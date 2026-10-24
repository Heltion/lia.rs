//! Register-resident online softmax shared by prefill and compact GQA decode.
use super::llama_flash_math::{Float2, exp, load_queries, load_tile, pair, pv, pv_f32, qk};
use super::llama_flash_register::pv_register;
use cuda_device::{device, thread, warp};
#[device]
pub(super) unsafe fn attention<
    const ROWS: usize,
    const THREADS: usize,
    const COLUMNS: usize,
    const KV_HEADS: usize,
    const GROUP: usize,
    const FULL_P: bool,
    const TENSOR_PV: bool,
    const REGISTER_P: bool,
>(
    query_start: usize,
    query_count: usize,
    position: usize,
    parts: usize,
    span: usize,
    paged: bool,
    stride: usize,
    positions: &[u32],
    sequences: &[u32],
    pages: &[u32],
    q: &[f32],
    k: &[u16],
    v: &[u16],
    normalize: bool,
    partial: *mut f32,
) {
    let tid = thread::index_1d().get();
    let local = tid % THREADS;
    let lane = local % 32;
    let warp_id = local / 32;
    let block = tid / THREADS;
    let part = block % parts;
    let head = block / parts % KV_HEADS;
    let rows = if ROWS == 16 { GROUP } else { ROWS };
    let query_base = block / parts / KV_HEADS * rows + query_start * GROUP;
    let queries = ((query_start + query_count) * GROUP).min(if ROWS == 16 {
        query_base + GROUP
    } else {
        usize::MAX
    });
    let shared = if COLUMNS == 64 {
        static mut WIDE: cuda_device::SharedArray<u32, 11008, 16> =
            cuda_device::SharedArray::UNINIT;
        unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut WIDE) }
    } else if ROWS == 64 {
        static mut PREFILL: cuda_device::SharedArray<u32, 7808, 16> =
            cuda_device::SharedArray::UNINIT;
        unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut PREFILL) }
    } else {
        static mut DECODE: cuda_device::SharedArray<u32, 3584, 16> =
            cuda_device::SharedArray::UNINIT;
        unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut DECODE) }
    };
    let query = shared;
    let kv = unsafe { shared.add(ROWS * 68) };
    let probability = if FULL_P {
        query
    } else {
        unsafe { shared.add(ROWS * 68 + COLUMNS * 68) }
    };
    let mut maxima = [f32::NEG_INFINITY; 2];
    let mut totals = [0_f32; 2];
    let mut output = [[0_f32; 4]; 16];
    unsafe {
        load_queries::<ROWS, THREADS, KV_HEADS, GROUP>(q, query, query_base, queries, head, local);
    }
    thread::sync_threads();
    let last = ((query_base + ROWS).min(queries) - 1) / GROUP;
    let maximum_keys = if paged {
        unsafe { *positions.as_ptr().add(last) as usize + 1 }
    } else {
        position + last + 1
    };
    let sequence = if paged {
        unsafe { *sequences.as_ptr().add(query_base / GROUP) as usize }
    } else {
        0
    };
    let end = ((part + 1) * span).min(maximum_keys);
    let mut start = part * span;
    while start < end {
        let physical = if paged {
            unsafe { *pages.as_ptr().add(sequence * stride + start / 32) as usize * 32 }
        } else {
            start
        };
        unsafe {
            load_tile::<true, THREADS, KV_HEADS>(k, kv, physical, head, start, end, local);
        }
        let physical_b = if COLUMNS == 64 && start + 32 < end {
            if paged {
                unsafe { *pages.as_ptr().add(sequence * stride + (start + 32) / 32) as usize * 32 }
            } else {
                start + 32
            }
        } else {
            0
        };
        if COLUMNS == 64 {
            unsafe {
                load_tile::<true, THREADS, KV_HEADS>(
                    k,
                    kv.add(32 * 68),
                    physical_b,
                    head,
                    start + 32,
                    end,
                    local,
                );
            }
        }
        unsafe {
            cuda_device::async_copy::cp_async_commit_group();
            cuda_device::async_copy::cp_async_wait_group(0);
        }
        thread::sync_threads();
        let mut scores = unsafe { qk::<COLUMNS>(query.add(warp_id * 16 * 68), kv, lane as u32) };
        // QK has finished reading K; the same tile receives V while softmax runs.
        thread::sync_threads();
        unsafe {
            load_tile::<true, THREADS, KV_HEADS>(v, kv, physical, head, start, end, local);
            if COLUMNS == 64 {
                load_tile::<true, THREADS, KV_HEADS>(
                    v,
                    kv.add(32 * 68),
                    physical_b,
                    head,
                    start + 32,
                    end,
                    local,
                );
            }
            cuda_device::async_copy::cp_async_commit_group();
        }
        let mut factors = [0_f32; 2];
        macro_rules! softmax_row {
            ($r:literal) => {{
                let row = warp_id * 16 + lane / 4 + $r * 8;
                let at = query_base + row;
                let last_key = if at < queries {
                    if paged {
                        unsafe { *positions.as_ptr().add(at / GROUP) as usize + 1 }
                    } else {
                        position + at / GROUP + 1
                    }
                } else {
                    0
                };
                let limit = end.min(last_key);
                let mut maximum = f32::NEG_INFINITY;
                macro_rules! mask_pair {
                    ($j:literal) => {{
                        let column = $j * 8 + lane % 4 * 2;
                        let a = if at < queries && start + column < limit {
                            scores[$j][$r * 2] * (core::f32::consts::LOG2_E * 0.08838834764831845)
                        } else {
                            f32::NEG_INFINITY
                        };
                        let b = if at < queries && start + column + 1 < limit {
                            scores[$j][$r * 2 + 1]
                                * (core::f32::consts::LOG2_E * 0.08838834764831845)
                        } else {
                            f32::NEG_INFINITY
                        };
                        scores[$j][$r * 2] = a;
                        scores[$j][$r * 2 + 1] = b;
                        maximum = maximum.max(a).max(b);
                    }};
                }
                mask_pair!(0);
                mask_pair!(1);
                mask_pair!(2);
                mask_pair!(3);
                if COLUMNS == 64 {
                    mask_pair!(4);
                    mask_pair!(5);
                    mask_pair!(6);
                    mask_pair!(7);
                }
                maximum = maximum.max(warp::shuffle_f32_sync(
                    0xffff_ffff,
                    maximum,
                    (lane ^ 1) as u32,
                ));
                maximum = maximum.max(warp::shuffle_f32_sync(
                    0xffff_ffff,
                    maximum,
                    (lane ^ 2) as u32,
                ));
                maximum = maximum.max(maxima[$r]);
                factors[$r] = if maxima[$r] == f32::NEG_INFINITY {
                    0.0
                } else {
                    exp(maxima[$r] - maximum)
                };
                let mut total = 0_f32;
                macro_rules! probability_pair {
                    ($j:literal) => {{
                        let a = scores[$j][$r * 2];
                        let b = scores[$j][$r * 2 + 1];
                        let a = if a == f32::NEG_INFINITY {
                            0.0
                        } else {
                            exp(a - maximum)
                        };
                        let b = if b == f32::NEG_INFINITY {
                            0.0
                        } else {
                            exp(b - maximum)
                        };
                        total += a;
                        total += b;
                        unsafe {
                            if REGISTER_P {
                                scores[$j][$r * 2] = a * 32768.0;
                                scores[$j][$r * 2 + 1] = b * 32768.0;
                            } else if TENSOR_PV {
                                let a = a * 32768.0;
                                let b = b * 32768.0;
                                let ah = (a as f16) as f32;
                                let bh = (b as f16) as f32;
                                let at = row * ((COLUMNS + 8) / 2) + $j * 4 + lane % 4;
                                *probability.add(at) = pair(ah, bh);
                                *probability.add(ROWS * ((COLUMNS + 8) / 2) + at) =
                                    pair(a - ah, b - bh);
                            } else if FULL_P {
                                probability
                                    .add(row * COLUMNS + $j * 8 + lane % 4 * 2)
                                    .cast::<Float2>()
                                    .write(Float2 { values: [a, b] });
                            } else {
                                *probability.add(row * ((COLUMNS + 8) / 2) + $j * 4 + lane % 4) =
                                    pair(a, b);
                            }
                        }
                    }};
                }
                probability_pair!(0);
                probability_pair!(1);
                probability_pair!(2);
                probability_pair!(3);
                if COLUMNS == 64 {
                    probability_pair!(4);
                    probability_pair!(5);
                    probability_pair!(6);
                    probability_pair!(7);
                }
                total += warp::shuffle_f32_sync(0xffff_ffff, total, (lane ^ 1) as u32);
                total += warp::shuffle_f32_sync(0xffff_ffff, total, (lane ^ 2) as u32);
                totals[$r] =
                    totals[$r] * factors[$r] + total * if TENSOR_PV { 32768.0 } else { 1.0 };
                maxima[$r] = maximum;
            }};
        }
        softmax_row!(0);
        softmax_row!(1);
        unsafe {
            cuda_device::async_copy::cp_async_wait_group(0);
        }
        thread::sync_threads();
        macro_rules! scale {
            ($j:literal) => {{
                output[$j][0] *= factors[0];
                output[$j][1] *= factors[0];
                output[$j][2] *= factors[1];
                output[$j][3] *= factors[1];
            }};
        }
        scale!(0);
        scale!(1);
        scale!(2);
        scale!(3);
        scale!(4);
        scale!(5);
        scale!(6);
        scale!(7);
        scale!(8);
        scale!(9);
        scale!(10);
        scale!(11);
        scale!(12);
        scale!(13);
        scale!(14);
        scale!(15);
        output = unsafe {
            if REGISTER_P {
                pv_register::<COLUMNS>(output, scores, kv, lane as u32)
            } else if TENSOR_PV {
                let high = probability.add(warp_id * 16 * ((COLUMNS + 8) / 2));
                let low = high.add(ROWS * ((COLUMNS + 8) / 2));
                let next = pv::<COLUMNS>(output, high, kv, lane as u32);
                pv::<COLUMNS>(next, low, kv, lane as u32)
            } else if FULL_P {
                pv_f32::<COLUMNS>(output, probability.cast(), kv, warp_id, lane)
            } else {
                pv::<COLUMNS>(
                    output,
                    probability.add(warp_id * 16 * ((COLUMNS + 8) / 2)),
                    kv,
                    lane as u32,
                )
            }
        };
        // All probability readers must finish before the query tile is restored.
        thread::sync_threads();
        if FULL_P && !REGISTER_P && start + COLUMNS < end {
            unsafe {
                load_queries::<ROWS, THREADS, KV_HEADS, GROUP>(
                    q, query, query_base, queries, head, local,
                );
            }
            thread::sync_threads();
        }
        start += COLUMNS;
    }
    macro_rules! write_row {
        ($r:literal) => {{
            let at = query_base + warp_id * 16 + lane / 4 + $r * 8;
            if at < queries {
                let qhead = head * GROUP + at % GROUP;
                let base = if normalize {
                    (at / GROUP * (KV_HEADS * GROUP) + qhead) * 128
                } else {
                    (((at / GROUP - query_start) * (KV_HEADS * GROUP) + qhead) * parts + part) * 130
                };
                macro_rules! write_pair {
                    ($j:literal) => {{
                        let column = $j * 8 + lane % 4 * 2;
                        unsafe {
                            partial.add(base + column).cast::<Float2>().write(Float2 {
                                values: if normalize {
                                    [
                                        output[$j][$r * 2] / totals[$r],
                                        output[$j][$r * 2 + 1] / totals[$r],
                                    ]
                                } else {
                                    [output[$j][$r * 2], output[$j][$r * 2 + 1]]
                                },
                            });
                        }
                    }};
                }
                write_pair!(0);
                write_pair!(1);
                write_pair!(2);
                write_pair!(3);
                write_pair!(4);
                write_pair!(5);
                write_pair!(6);
                write_pair!(7);
                write_pair!(8);
                write_pair!(9);
                write_pair!(10);
                write_pair!(11);
                write_pair!(12);
                write_pair!(13);
                write_pair!(14);
                write_pair!(15);
                if !normalize && lane.is_multiple_of(4) {
                    unsafe {
                        partial.add(base + 128).cast::<Float2>().write(Float2 {
                            // Merge also consumes scalar-attention partials, whose
                            // maxima use natural logarithms. Convert only at this boundary.
                            values: [maxima[$r] * core::f32::consts::LN_2, totals[$r]],
                        });
                    }
                }
            }
        }};
    }
    write_row!(0);
    write_row!(1);
}
