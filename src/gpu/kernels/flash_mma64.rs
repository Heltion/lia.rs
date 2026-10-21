//! Sixty-four queries share KV tiles and reuse score storage for probabilities.
use cuda_device::{kernel, launch_bounds, launch_contract, thread, warp, wmma};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;

    use super::super::flash_mma_math::{Float2, Words4, exp, multiply, split};

    #[kernel]
    #[launch_bounds(1024)]
    #[launch_contract(domain = 1, dynamic_shared_range = (0, 96256), dynamic_shared_alignment = 16)]
    pub unsafe fn flash_mma64<const GROUPED: bool>(
        q_heads: usize,
        kv_heads: usize,
        tokens: usize,
        span: usize,
        parts: usize,
        query_start: usize,
        query_count: usize,
        q: &[f32],
        k: &[u16],
        v: &[u16],
        partial: *mut f32,
    ) {
        const ROWS: u32 = 64;
        let tid = thread::index_1d().get() as u32;
        let block = ROWS * 16;
        let local = tid % block;
        let lane = local % 32;
        let warp_id = local / 32;
        let qh = q_heads as u32;
        let kvh = kv_heads as u32;
        let parts = parts as u32;
        let tokens = tokens as u32;
        let span = span as u32;
        let batch = q.len() as u32 / (qh * 256);
        let part = tid / block % parts;
        let groups = if GROUPED { qh / kvh } else { 1 };
        let heads = if GROUPED { kvh } else { qh };
        let head = tid / block / parts % heads;
        let query_base = tid / block / parts / heads * ROWS + query_start as u32 * groups;
        let kh = if GROUPED { head } else { head / (qh / kvh) };
        let queries = (query_start as u32 + query_count as u32) * groups;
        // Layouts are padded to preserve 16-byte matrix-load alignment. All
        // CTAs are complete and every thread reaches each block barrier.
        let tile = cuda_device::DynamicSharedArray::<u32, 16>::get();
        let qhi = tile;
        let qlo = unsafe { tile.add((ROWS * 132) as usize) };
        let kv = unsafe { tile.add((ROWS * 264) as usize) };
        let phi = unsafe { tile.add((ROWS * 264 + 4224) as usize) };
        let plo = unsafe { phi.add(20) };
        let scores = phi.cast::<f32>();
        let metadata = unsafe { phi.add((ROWS * 44) as usize).cast::<f32>() };
        // Each row owns 44 words: its QK scores are consumed before the same
        // warp writes its two probability fragments. Reduction shuffles ensure
        // every lane has loaded its score before any overlapping store. Row
        // padding preserves matrix-load alignment and bank distribution.
        let mut maxima = [f32::NEG_INFINITY; 2];
        let mut totals = [0.0f32; 2];
        let mut output = [[0.0f32; 4]; 4];
        let mut i = local;
        while i < ROWS * 128 {
            let row = i / 128;
            let col = i % 128 * 2;
            let query = query_base + row;
            let (a, b) = if query < queries {
                let query_head = if GROUPED {
                    kh * groups + query % groups
                } else {
                    head
                };
                let at = (((query / groups) * qh + query_head) * 256 + col) as usize;
                unsafe {
                    let pair = q.as_ptr().add(at).cast::<Float2>().read();
                    (pair.values[0], pair.values[1])
                }
            } else {
                (0.0, 0.0)
            };
            let at = (row * 132 + col / 2) as usize;
            unsafe {
                split(qhi.add(at), qlo.add(at), a, b);
            }
            i += block;
        }
        thread::sync_threads();
        let end = ((part + 1) * span)
            .min(tokens)
            .min(tokens - batch + ((query_base + ROWS).min(queries) - 1) / groups + 1);
        let mut start = part * span;
        while start < end {
            // Both the 128-word source rows and 132-word shared rows are
            // aligned for four contiguous packed words.
            i = local * 4;
            while i < 32 * 128 {
                let token = start + i / 128;
                let word = if token < tokens {
                    let at = ((token * kvh + kh) * 128 + i % 128) as usize;
                    unsafe { k.as_ptr().cast::<u32>().add(at).cast::<Words4>().read() }
                } else {
                    Words4 { words: [0; 4] }
                };
                unsafe {
                    kv.add((i / 128 * 132 + i % 128) as usize)
                        .cast::<Words4>()
                        .write(word);
                }
                i += block * 4;
            }
            thread::sync_threads();
            if warp_id < ROWS / 4 {
                let s = unsafe {
                    multiply::<256, 264, false>(
                        [0.0; 4],
                        qhi.add((warp_id / 4 * 16 * 132) as usize),
                        qlo.add((warp_id / 4 * 16 * 132) as usize),
                        kv,
                        warp_id % 4 * 8,
                        lane,
                    )
                };
                let row = lane / 4 + warp_id / 4 * 16;
                let col = warp_id % 4 * 8 + lane % 4 * 2;
                unsafe {
                    *scores.add((row * 44 + col) as usize) = s[0] / 16.0;
                    *scores.add((row * 44 + col + 1) as usize) = s[1] / 16.0;
                    *scores.add(((row + 8) * 44 + col) as usize) = s[2] / 16.0;
                    *scores.add(((row + 8) * 44 + col + 1) as usize) = s[3] / 16.0;
                }
            }
            thread::sync_threads();
            {
                // QK no longer reads kv. Stage V while the unchanged softmax
                // runs, without reserving another shared-memory tile.
                i = local * 4;
                while i < 32 * 128 {
                    let token = start + i / 128;
                    let dest = unsafe { kv.add((i / 128 * 132 + i % 128) as usize) };
                    if token < tokens {
                        let at = ((token * kvh + kh) * 128 + i % 128) as usize;
                        unsafe {
                            cuda_device::async_copy::cp_async_ca_16(
                                dest,
                                v.as_ptr().cast::<u32>().add(at),
                            );
                        }
                    } else {
                        unsafe { dest.cast::<Words4>().write(Words4 { words: [0; 4] }) };
                    }
                    i += block * 4;
                }
                unsafe { cuda_device::async_copy::cp_async_commit_group() };
            }
            macro_rules! row_pair {
                ($r:literal) => {{
                    let r = $r;

                    let row = warp_id + r * (ROWS / 2);
                    let valid = query_base + row < queries
                        && start + lane < tokens - batch + (query_base + row) / groups + 1
                        && start + lane < tokens;
                    let score = if valid {
                        unsafe { *scores.add((row * 44 + lane) as usize) }
                    } else {
                        f32::NEG_INFINITY
                    };
                    let mut maximum = score;
                    let mut d = 16;
                    while d > 0 {
                        maximum = maximum.max(warp::shuffle_down_f32_sync(0xffff_ffff, maximum, d));
                        d /= 2;
                    }
                    maximum =
                        warp::shuffle_f32_sync(0xffff_ffff, maximum, 0).max(maxima[r as usize]);
                    let factor = if maxima[r as usize] == f32::NEG_INFINITY {
                        0.0
                    } else {
                        exp(maxima[r as usize] - maximum)
                    };
                    let probability = if valid { exp(score - maximum) } else { 0.0 };
                    let mut total = probability;
                    d = 16;
                    while d > 0 {
                        total += warp::shuffle_down_f32_sync(0xffff_ffff, total, d);
                        d /= 2;
                    }
                    totals[r as usize] =
                        totals[r as usize] * factor + warp::shuffle_f32_sync(0xffff_ffff, total, 0);
                    maxima[r as usize] = maximum;
                    let next = warp::shuffle_down_f32_sync(0xffff_ffff, probability, 1);
                    if lane.is_multiple_of(2) {
                        let at = (row * 44 + lane / 2) as usize;
                        unsafe {
                            split(phi.add(at), plo.add(at), probability, next);
                        }
                    }
                    // Each output fragment owns rows lane/4 and lane/4+8.
                    if lane == 0 {
                        unsafe {
                            *metadata.add(row as usize) = factor;
                        }
                    }
                }};
            }
            row_pair!(0);
            row_pair!(1);
            {
                // Wait for every issuing thread's copies before the block-wide
                // barrier makes V and the probability fragments visible.
                unsafe { cuda_device::async_copy::cp_async_wait_group(0) };
            }
            thread::sync_threads();
            let a = unsafe { *metadata.add((warp_id / 8 * 16 + lane / 4) as usize) };
            let b = unsafe { *metadata.add((warp_id / 8 * 16 + 8 + lane / 4) as usize) };
            {
                let mut j = 0;
                #[unroll]
                while j < 4 {
                    let old = output[j];
                    output[j] = [old[0] * a, old[1] * a, old[2] * b, old[3] * b];
                    j += 1;
                }
                let mut k = 0;
                while k < 32 {
                    let offset =
                        (warp_id / 8 * 16 * 44 + (lane % 16 * 88 + k + lane / 16 * 8) / 2) as usize;
                    let ah = unsafe { wmma::ldmatrix_x4(phi.add(offset)) };
                    let al = unsafe { wmma::ldmatrix_x4(plo.add(offset)) };
                    j = 0;
                    #[unroll]
                    while j < 4 {
                        let col = warp_id % 8 * 32 + j as u32 * 8;
                        let bv = unsafe {
                            wmma::ldmatrix_x2_trans(
                                kv.add(((k + lane % 16) * 132 + col / 2) as usize),
                            )
                        };
                        // Each fragment retains the same low/high and K order.
                        output[j] = unsafe { wmma::mma_m16n8k16_f32_f16(output[j], al, bv) };
                        output[j] = unsafe { wmma::mma_m16n8k16_f32_f16(output[j], ah, bv) };
                        j += 1;
                    }
                    k += 16;
                }
            }
            thread::sync_threads();
            start += 32;
        }
        // Convert warp-owned row metadata into a shared table for all fragments.
        if lane == 0 {
            macro_rules! row_pair {
                ($r:literal) => {{
                    let r = $r;

                    unsafe {
                        *metadata.add((warp_id + r * (ROWS / 2)) as usize) = maxima[r as usize];
                        *metadata.add((ROWS + warp_id + r * (ROWS / 2)) as usize) =
                            totals[r as usize];
                    }
                }};
            }
            row_pair!(0);
            row_pair!(1);
        }
        thread::sync_threads();
        let mut j = 0;
        #[unroll]
        while j < 4 {
            let col = warp_id % 8 * 32 + j as u32 * 8 + lane % 4 * 2;
            macro_rules! row_pair {
                ($r:literal) => {{
                    let r = $r;

                    let row = lane / 4 + r * 8 + warp_id / 8 * 16;
                    let query = query_base + row;
                    if query < queries {
                        let query_head = if GROUPED {
                            kh * groups + query % groups
                        } else {
                            head
                        };
                        let base = (((((query / groups - query_start as u32) * qh + query_head)
                            * parts)
                            + part)
                            * 258) as usize;
                        unsafe {
                            partial
                                .add(base + col as usize)
                                .cast::<Float2>()
                                .write(Float2 {
                                    values: [
                                        output[j][r as usize * 2],
                                        output[j][r as usize * 2 + 1],
                                    ],
                                });
                            if col == 0 {
                                partial.add(base + 256).cast::<Float2>().write(Float2 {
                                    values: [
                                        *metadata.add(row as usize),
                                        *metadata.add((ROWS + row) as usize),
                                    ],
                                });
                            }
                        }
                    }
                }};
            }
            row_pair!(0);
            row_pair!(1);
            j += 1;
        }
    }
}
