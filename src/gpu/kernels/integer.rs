use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub unsafe fn quantize_activation(input: &[f32], packed: *mut u32, scales: *mut f32) {
        let i = thread::index_1d().get();
        if i < input.len() {
            let lane = (i % 32) as u32;
            let mut max = input[i].abs();
            let mut delta = 16;
            while delta > 0 {
                max = max.max(cuda_device::warp::shuffle_down_f32_sync(
                    0xffff_ffff,
                    max,
                    delta,
                ));
                delta /= 2;
            }
            max = cuda_device::warp::shuffle_f32_sync(0xffff_ffff, max, 0);
            let scale = if max == 0.0 { 1.0 } else { max / 127.0 };
            let q = (input[i] / scale).round_ties_even() as i32 as u32;
            let base = lane / 4 * 4;
            let q0 = cuda_device::warp::shuffle_sync(0xffff_ffff, q, base) & 255;
            let q1 = cuda_device::warp::shuffle_sync(0xffff_ffff, q, base + 1) & 255;
            let q2 = cuda_device::warp::shuffle_sync(0xffff_ffff, q, base + 2) & 255;
            let q3 = cuda_device::warp::shuffle_sync(0xffff_ffff, q, base + 3) & 255;
            unsafe {
                if lane.is_multiple_of(4) {
                    *packed.add(i / 4) = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
                }
                if lane == 0 {
                    *scales.add(i / 32) = scale;
                }
            }
        }
    }

    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    struct Words4 {
        data: [u32; 4],
    }

    #[inline(always)]
    fn int_dot16<const KIND: u32>(
        w0: u32,
        w1: u32,
        w2: u32,
        w3: u32,
        q0: u32,
        q1: u32,
        q2: u32,
        q3: u32,
        scale: f32,
        offset0: f32,
        offset1: f32,
    ) -> f32 {
        let dot = cuda_device::dotprod::dp4a_s32(w0, q0, 0);
        let dot = cuda_device::dotprod::dp4a_s32(w1, q1, dot);
        let dot = cuda_device::dotprod::dp4a_s32(w2, q2, dot);
        let dot = cuda_device::dotprod::dp4a_s32(w3, q3, dot) as f32;
        if KIND == 10 || KIND == 12 || KIND == 19 || KIND == 29 {
            let a = cuda_device::dotprod::dp4a_s32(0x01010101, q0, 0);
            let a = cuda_device::dotprod::dp4a_s32(0x01010101, q1, a);
            let b = cuda_device::dotprod::dp4a_s32(0x01010101, q2, 0);
            let b = cuda_device::dotprod::dp4a_s32(0x01010101, q3, b);
            if KIND == 29 {
                scale * dot + offset0 * a as f32 + offset1 * b as f32
            } else {
                scale * dot + offset0 * (a + b) as f32
            }
        } else {
            scale * dot
        }
    }

    // Explicit scalar accumulators avoid dynamically indexed local arrays in
    // this hot kernel. Keep its reduction and split-row writes together even
    // though this function exceeds the usual file-size guideline.
    #[kernel]
    pub unsafe fn matmul_int<
        const KIND: u32,
        const BYTES: usize,
        const MAX: usize,
        const SPLIT: usize,
        const ROW_GROUP: usize,
    >(
        cols: usize,
        rows: usize,
        batch: usize,
        weights: &[u8],
        x: &[u32],
        scales: &[f32],
        out: *mut f32,
    ) {
        static mut PARTIAL: cuda_device::SharedArray<f32, 64> = cuda_device::SharedArray::UNINIT;
        let tid = thread::index_1d().get() as u32;
        let row = (tid / (32 * SPLIT as u32)) * ROW_GROUP as u32;
        let split = (tid / 32) % SPLIT as u32;
        let lane = tid % 32;
        let cols = cols as u32;
        let rows = rows as u32;
        let batch = batch as u32;
        if row < rows {
            let mut sum0 = 0.0f32;
            let mut sum1 = 0.0f32;
            let mut sum2 = 0.0f32;
            let mut sum3 = 0.0f32;
            let mut sum4 = 0.0f32;
            let mut sum5 = 0.0f32;
            let mut sum6 = 0.0f32;
            let mut sum7 = 0.0f32;
            let mut sum0_b = 0.0f32;
            let mut sum1_b = 0.0f32;
            let mut sum2_b = 0.0f32;
            let mut sum3_b = 0.0f32;
            let mut sum4_b = 0.0f32;
            let mut sum5_b = 0.0f32;
            let mut sum6_b = 0.0f32;
            let mut sum7_b = 0.0f32;
            let mut col = (split * 32 + lane) * 16;
            while col < cols {
                let block = (row * cols + col) / 256;
                let data = unsafe {
                    core::slice::from_raw_parts(
                        weights.as_ptr().add((block * BYTES as u32) as usize),
                        BYTES,
                    )
                };
                let i = (col & 240) as usize;
                let (w0, w1, scale, offset0) =
                    unsafe { crate::quant::packed_eight_aligned(KIND, data, i) };
                let (w2, w3, _, offset1) =
                    unsafe { crate::quant::packed_eight_aligned(KIND, data, i + 8) };
                let (v0, v1, v2, v3, scale_b, offset0_b, offset1_b) =
                    if ROW_GROUP == 2 && row + 1 < rows {
                        let block = ((row + 1) * cols + col) / 256;
                        let data = unsafe {
                            core::slice::from_raw_parts(
                                weights.as_ptr().add((block * BYTES as u32) as usize),
                                BYTES,
                            )
                        };
                        let (a, b, scale, off0) =
                            unsafe { crate::quant::packed_eight_aligned(KIND, data, i) };
                        let (c, d, _, off1) =
                            unsafe { crate::quant::packed_eight_aligned(KIND, data, i + 8) };
                        (a, b, c, d, scale, off0, off1)
                    } else {
                        (0, 0, 0, 0, 0.0, 0.0, 0.0)
                    };
                if MAX > 0 && batch > 0 {
                    let base = (col) / 4;
                    // SAFETY: the host validates 16-byte alignment; columns
                    // and each vector offset cover complete four-word groups.
                    let [q0, q1, q2, q3] =
                        unsafe { x.as_ptr().add(base as usize).cast::<Words4>().read().data };
                    let xs = unsafe { *scales.get_unchecked(((col) / 32) as usize) };
                    sum0 += xs
                        * int_dot16::<KIND>(
                            w0, w1, w2, w3, q0, q1, q2, q3, scale, offset0, offset1,
                        );
                    if ROW_GROUP == 2 && row + 1 < rows {
                        sum0_b += xs
                            * int_dot16::<KIND>(
                                v0, v1, v2, v3, q0, q1, q2, q3, scale_b, offset0_b, offset1_b,
                            );
                    }
                }
                if MAX > 1 && batch > 1 {
                    let base = (cols + col) / 4;
                    // SAFETY: the host validates 16-byte alignment; columns
                    // and each vector offset cover complete four-word groups.
                    let [q0, q1, q2, q3] =
                        unsafe { x.as_ptr().add(base as usize).cast::<Words4>().read().data };
                    let xs = unsafe { *scales.get_unchecked(((cols + col) / 32) as usize) };
                    sum1 += xs
                        * int_dot16::<KIND>(
                            w0, w1, w2, w3, q0, q1, q2, q3, scale, offset0, offset1,
                        );
                    if ROW_GROUP == 2 && row + 1 < rows {
                        sum1_b += xs
                            * int_dot16::<KIND>(
                                v0, v1, v2, v3, q0, q1, q2, q3, scale_b, offset0_b, offset1_b,
                            );
                    }
                }
                if MAX > 2 && batch > 2 {
                    let base = (2 * cols + col) / 4;
                    // SAFETY: the host validates 16-byte alignment; columns
                    // and each vector offset cover complete four-word groups.
                    let [q0, q1, q2, q3] =
                        unsafe { x.as_ptr().add(base as usize).cast::<Words4>().read().data };
                    let xs = unsafe { *scales.get_unchecked(((2 * cols + col) / 32) as usize) };
                    sum2 += xs
                        * int_dot16::<KIND>(
                            w0, w1, w2, w3, q0, q1, q2, q3, scale, offset0, offset1,
                        );
                    if ROW_GROUP == 2 && row + 1 < rows {
                        sum2_b += xs
                            * int_dot16::<KIND>(
                                v0, v1, v2, v3, q0, q1, q2, q3, scale_b, offset0_b, offset1_b,
                            );
                    }
                }
                if MAX > 3 && batch > 3 {
                    let base = (3 * cols + col) / 4;
                    // SAFETY: the host validates 16-byte alignment; columns
                    // and each vector offset cover complete four-word groups.
                    let [q0, q1, q2, q3] =
                        unsafe { x.as_ptr().add(base as usize).cast::<Words4>().read().data };
                    let xs = unsafe { *scales.get_unchecked(((3 * cols + col) / 32) as usize) };
                    sum3 += xs
                        * int_dot16::<KIND>(
                            w0, w1, w2, w3, q0, q1, q2, q3, scale, offset0, offset1,
                        );
                    if ROW_GROUP == 2 && row + 1 < rows {
                        sum3_b += xs
                            * int_dot16::<KIND>(
                                v0, v1, v2, v3, q0, q1, q2, q3, scale_b, offset0_b, offset1_b,
                            );
                    }
                }
                if MAX > 4 && batch > 4 {
                    let base = (4 * cols + col) / 4;
                    // SAFETY: the host validates 16-byte alignment; columns
                    // and each vector offset cover complete four-word groups.
                    let [q0, q1, q2, q3] =
                        unsafe { x.as_ptr().add(base as usize).cast::<Words4>().read().data };
                    let xs = unsafe { *scales.get_unchecked(((4 * cols + col) / 32) as usize) };
                    sum4 += xs
                        * int_dot16::<KIND>(
                            w0, w1, w2, w3, q0, q1, q2, q3, scale, offset0, offset1,
                        );
                    if ROW_GROUP == 2 && row + 1 < rows {
                        sum4_b += xs
                            * int_dot16::<KIND>(
                                v0, v1, v2, v3, q0, q1, q2, q3, scale_b, offset0_b, offset1_b,
                            );
                    }
                }
                if MAX > 5 && batch > 5 {
                    let base = (5 * cols + col) / 4;
                    // SAFETY: the host validates 16-byte alignment; columns
                    // and each vector offset cover complete four-word groups.
                    let [q0, q1, q2, q3] =
                        unsafe { x.as_ptr().add(base as usize).cast::<Words4>().read().data };
                    let xs = unsafe { *scales.get_unchecked(((5 * cols + col) / 32) as usize) };
                    sum5 += xs
                        * int_dot16::<KIND>(
                            w0, w1, w2, w3, q0, q1, q2, q3, scale, offset0, offset1,
                        );
                    if ROW_GROUP == 2 && row + 1 < rows {
                        sum5_b += xs
                            * int_dot16::<KIND>(
                                v0, v1, v2, v3, q0, q1, q2, q3, scale_b, offset0_b, offset1_b,
                            );
                    }
                }
                if MAX > 6 && batch > 6 {
                    let base = (6 * cols + col) / 4;
                    // SAFETY: the host validates 16-byte alignment; columns
                    // and each vector offset cover complete four-word groups.
                    let [q0, q1, q2, q3] =
                        unsafe { x.as_ptr().add(base as usize).cast::<Words4>().read().data };
                    let xs = unsafe { *scales.get_unchecked(((6 * cols + col) / 32) as usize) };
                    sum6 += xs
                        * int_dot16::<KIND>(
                            w0, w1, w2, w3, q0, q1, q2, q3, scale, offset0, offset1,
                        );
                    if ROW_GROUP == 2 && row + 1 < rows {
                        sum6_b += xs
                            * int_dot16::<KIND>(
                                v0, v1, v2, v3, q0, q1, q2, q3, scale_b, offset0_b, offset1_b,
                            );
                    }
                }
                if MAX > 7 && batch > 7 {
                    let base = (7 * cols + col) / 4;
                    // SAFETY: the host validates 16-byte alignment; columns
                    // and each vector offset cover complete four-word groups.
                    let [q0, q1, q2, q3] =
                        unsafe { x.as_ptr().add(base as usize).cast::<Words4>().read().data };
                    let xs = unsafe { *scales.get_unchecked(((7 * cols + col) / 32) as usize) };
                    sum7 += xs
                        * int_dot16::<KIND>(
                            w0, w1, w2, w3, q0, q1, q2, q3, scale, offset0, offset1,
                        );
                    if ROW_GROUP == 2 && row + 1 < rows {
                        sum7_b += xs
                            * int_dot16::<KIND>(
                                v0, v1, v2, v3, q0, q1, q2, q3, scale_b, offset0_b, offset1_b,
                            );
                    }
                }
                col += 512 * SPLIT as u32;
            }
            let mut delta = 16;
            while delta > 0 {
                if MAX > 0 {
                    sum0 += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum0, delta);
                    sum0_b += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum0_b, delta);
                }
                if MAX > 1 {
                    sum1 += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum1, delta);
                    sum1_b += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum1_b, delta);
                }
                if MAX > 2 {
                    sum2 += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum2, delta);
                    sum2_b += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum2_b, delta);
                }
                if MAX > 3 {
                    sum3 += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum3, delta);
                    sum3_b += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum3_b, delta);
                }
                if MAX > 4 {
                    sum4 += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum4, delta);
                    sum4_b += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum4_b, delta);
                }
                if MAX > 5 {
                    sum5 += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum5, delta);
                    sum5_b += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum5_b, delta);
                }
                if MAX > 6 {
                    sum6 += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum6, delta);
                    sum6_b += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum6_b, delta);
                }
                if MAX > 7 {
                    sum7 += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum7, delta);
                    sum7_b += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum7_b, delta);
                }
                delta /= 2;
            }
            if SPLIT > 1 {
                // SAFETY: one complete CTA owns this row, with one writer per
                // warp and token. All threads participate in the barrier.
                let p = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut PARTIAL) };
                if lane == 0 {
                    unsafe {
                        if MAX > 0 && batch > 0 {
                            *p.add(split as usize * 16) = sum0;
                        }
                        if MAX > 1 && batch > 1 {
                            *p.add(split as usize * 16 + 1) = sum1;
                        }
                        if MAX > 2 && batch > 2 {
                            *p.add(split as usize * 16 + 2) = sum2;
                        }
                        if MAX > 3 && batch > 3 {
                            *p.add(split as usize * 16 + 3) = sum3;
                        }
                        if MAX > 4 && batch > 4 {
                            *p.add(split as usize * 16 + 4) = sum4;
                        }
                        if MAX > 5 && batch > 5 {
                            *p.add(split as usize * 16 + 5) = sum5;
                        }
                        if MAX > 6 && batch > 6 {
                            *p.add(split as usize * 16 + 6) = sum6;
                        }
                        if MAX > 7 && batch > 7 {
                            *p.add(split as usize * 16 + 7) = sum7;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 0 && batch > 0 {
                            *p.add(split as usize * 16 + 8) = sum0_b;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 1 && batch > 1 {
                            *p.add(split as usize * 16 + 9) = sum1_b;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 2 && batch > 2 {
                            *p.add(split as usize * 16 + 10) = sum2_b;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 3 && batch > 3 {
                            *p.add(split as usize * 16 + 11) = sum3_b;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 4 && batch > 4 {
                            *p.add(split as usize * 16 + 12) = sum4_b;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 5 && batch > 5 {
                            *p.add(split as usize * 16 + 13) = sum5_b;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 6 && batch > 6 {
                            *p.add(split as usize * 16 + 14) = sum6_b;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 7 && batch > 7 {
                            *p.add(split as usize * 16 + 15) = sum7_b;
                        }
                    }
                }
                thread::sync_threads();
                if split == 0 && lane == 0 {
                    unsafe {
                        if MAX > 0 && batch > 0 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16);
                                j += 1;
                            }
                            *out.add(row as usize) = total;
                        }
                        if MAX > 1 && batch > 1 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 1);
                                j += 1;
                            }
                            *out.add((rows + row) as usize) = total;
                        }
                        if MAX > 2 && batch > 2 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 2);
                                j += 1;
                            }
                            *out.add((2 * rows + row) as usize) = total;
                        }
                        if MAX > 3 && batch > 3 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 3);
                                j += 1;
                            }
                            *out.add((3 * rows + row) as usize) = total;
                        }
                        if MAX > 4 && batch > 4 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 4);
                                j += 1;
                            }
                            *out.add((4 * rows + row) as usize) = total;
                        }
                        if MAX > 5 && batch > 5 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 5);
                                j += 1;
                            }
                            *out.add((5 * rows + row) as usize) = total;
                        }
                        if MAX > 6 && batch > 6 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 6);
                                j += 1;
                            }
                            *out.add((6 * rows + row) as usize) = total;
                        }
                        if MAX > 7 && batch > 7 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 7);
                                j += 1;
                            }
                            *out.add((7 * rows + row) as usize) = total;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 0 && batch > 0 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 8);
                                j += 1;
                            }
                            *out.add((row + 1) as usize) = total;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 1 && batch > 1 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 9);
                                j += 1;
                            }
                            *out.add((rows + row + 1) as usize) = total;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 2 && batch > 2 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 10);
                                j += 1;
                            }
                            *out.add((2 * rows + row + 1) as usize) = total;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 3 && batch > 3 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 11);
                                j += 1;
                            }
                            *out.add((3 * rows + row + 1) as usize) = total;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 4 && batch > 4 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 12);
                                j += 1;
                            }
                            *out.add((4 * rows + row + 1) as usize) = total;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 5 && batch > 5 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 13);
                                j += 1;
                            }
                            *out.add((5 * rows + row + 1) as usize) = total;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 6 && batch > 6 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 14);
                                j += 1;
                            }
                            *out.add((6 * rows + row + 1) as usize) = total;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 7 && batch > 7 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 15);
                                j += 1;
                            }
                            *out.add((7 * rows + row + 1) as usize) = total;
                        }
                    }
                }
                return;
            }
            if lane == 0 {
                unsafe {
                    if MAX > 0 && batch > 0 {
                        *out.add((row) as usize) = sum0;
                    }
                    if MAX > 1 && batch > 1 {
                        *out.add((rows + row) as usize) = sum1;
                    }
                    if MAX > 2 && batch > 2 {
                        *out.add((2 * rows + row) as usize) = sum2;
                    }
                    if MAX > 3 && batch > 3 {
                        *out.add((3 * rows + row) as usize) = sum3;
                    }
                    if MAX > 4 && batch > 4 {
                        *out.add((4 * rows + row) as usize) = sum4;
                    }
                    if MAX > 5 && batch > 5 {
                        *out.add((5 * rows + row) as usize) = sum5;
                    }
                    if MAX > 6 && batch > 6 {
                        *out.add((6 * rows + row) as usize) = sum6;
                    }
                    if MAX > 7 && batch > 7 {
                        *out.add((7 * rows + row) as usize) = sum7;
                    }
                }
                // A paired row reuses the same activation vectors and scales.
                unsafe {
                    if ROW_GROUP == 2 && row + 1 < rows && MAX > 0 && batch > 0 {
                        *out.add((row + 1) as usize) = sum0_b;
                    }
                    if ROW_GROUP == 2 && row + 1 < rows && MAX > 1 && batch > 1 {
                        *out.add((rows + row + 1) as usize) = sum1_b;
                    }
                    if ROW_GROUP == 2 && row + 1 < rows && MAX > 2 && batch > 2 {
                        *out.add((2 * rows + row + 1) as usize) = sum2_b;
                    }
                    if ROW_GROUP == 2 && row + 1 < rows && MAX > 3 && batch > 3 {
                        *out.add((3 * rows + row + 1) as usize) = sum3_b;
                    }
                    if ROW_GROUP == 2 && row + 1 < rows && MAX > 4 && batch > 4 {
                        *out.add((4 * rows + row + 1) as usize) = sum4_b;
                    }
                    if ROW_GROUP == 2 && row + 1 < rows && MAX > 5 && batch > 5 {
                        *out.add((5 * rows + row + 1) as usize) = sum5_b;
                    }
                    if ROW_GROUP == 2 && row + 1 < rows && MAX > 6 && batch > 6 {
                        *out.add((6 * rows + row + 1) as usize) = sum6_b;
                    }
                    if ROW_GROUP == 2 && row + 1 < rows && MAX > 7 && batch > 7 {
                        *out.add((7 * rows + row + 1) as usize) = sum7_b;
                    }
                }
            }
        }
    }
}
