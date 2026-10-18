//! Sixteen causal queries share KV tiles; FP16 residuals preserve f32 queries.
use cuda_device::{convert, kernel, thread, warp, wmma};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[inline(always)]
    fn exp(x: f32) -> f32 {
        cuda_device::float::ex2_approx_f32(x * core::f32::consts::LOG2_E)
    }

    #[inline(always)]
    unsafe fn split(dest: *mut u32, low: *mut u32, a: f32, b: f32) {
        let hi = convert::cvt_f16x2_f32(a, b);
        let (ah, bh) = convert::cvt_f32x2_f16x2(hi);
        unsafe {
            dest.write(hi);
            low.write(convert::cvt_f16x2_f32(a - ah, b - bh));
        }
    }

    #[inline(always)]
    unsafe fn multiply<const K: u32, const STRIDE: u32, const TRANS: bool>(
        mut sum: [f32; 4],
        a: *const u32,
        low: *const u32,
        b: *const u32,
        col: u32,
        lane: u32,
    ) -> [f32; 4] {
        let mut k = 0;
        while k < K {
            unsafe {
                let offset = ((lane % 16 * STRIDE + k + lane / 16 * 8) / 2) as usize;
                let ah = wmma::ldmatrix_x4(a.add(offset));
                let al = wmma::ldmatrix_x4(low.add(offset));
                let bv = if TRANS {
                    wmma::ldmatrix_x2_trans(b.add(((k + lane % 16) * 132 + col / 2) as usize))
                } else {
                    wmma::ldmatrix_x2(
                        b.add(((col + lane % 8) * 132 + k / 2 + lane / 8 % 2 * 4) as usize),
                    )
                };
                sum = wmma::mma_m16n8k16_f32_f16(sum, al, bv);
                sum = wmma::mma_m16n8k16_f32_f16(sum, ah, bv);
            }
            k += 16;
        }
        sum
    }

    #[kernel]
    pub unsafe fn flash_mma<const GROUPED: bool>(
        q_heads: usize,
        kv_heads: usize,
        tokens: usize,
        span: usize,
        parts: usize,
        query_start: usize,
        q: &[f32],
        k: &[u16],
        v: &[u16],
        partial: *mut f32,
    ) {
        static mut TILE: cuda_device::SharedArray<u32, 9648, 16> = cuda_device::SharedArray::UNINIT;
        let tid = thread::index_1d().get() as u32;
        let local = tid % 256;
        let lane = local % 32;
        let warp_id = local / 32;
        let qh = q_heads as u32;
        let kvh = kv_heads as u32;
        let parts = parts as u32;
        let tokens = tokens as u32;
        let span = span as u32;
        let batch = q.len() as u32 / (qh * 256);
        let part = tid / 256 % parts;
        let groups = if GROUPED { qh / kvh } else { 1 };
        let heads = if GROUPED { kvh } else { qh };
        let head = tid / 256 / parts % heads;
        let query_base = tid / 256 / parts / heads * 16 + query_start as u32 * groups;
        let kh = if GROUPED { head } else { head / (qh / kvh) };
        let queries = batch * groups;
        // Layouts are padded to preserve 16-byte matrix-load alignment. All
        // CTAs are complete and every thread reaches each block barrier.
        let tile = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut TILE) };
        let qhi = tile;
        let qlo = unsafe { tile.add(2112) };
        let kv = unsafe { tile.add(4224) };
        let phi = unsafe { tile.add(8448) };
        let plo = unsafe { tile.add(8768) };
        let scores = unsafe { tile.add(9088).cast::<f32>() };
        // Metadata follows the 16-by-32 score tile.
        let mut maxima = [f32::NEG_INFINITY; 2];
        let mut totals = [0.0f32; 2];
        let mut output = [[0.0f32; 4]; 4];
        let mut i = local;
        while i < 16 * 128 {
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
                unsafe { (*q.get_unchecked(at), *q.get_unchecked(at + 1)) }
            } else {
                (0.0, 0.0)
            };
            let at = (row * 132 + col / 2) as usize;
            unsafe {
                split(qhi.add(at), qlo.add(at), a, b);
            }
            i += 256;
        }
        thread::sync_threads();
        let end = ((part + 1) * span)
            .min(tokens)
            .min(tokens - batch + ((query_base + 16).min(queries) - 1) / groups + 1);
        let mut start = part * span;
        while start < end {
            i = local;
            while i < 32 * 128 {
                let token = start + i / 128;
                let col = i % 128 * 2;
                let word = if token < tokens {
                    let at = ((token * kvh + kh) * 128 + col / 2) as usize;
                    unsafe { k.as_ptr().cast::<u32>().add(at).read() }
                } else {
                    0
                };
                unsafe {
                    kv.add((i / 128 * 132 + col / 2) as usize).write(word);
                }
                i += 256;
            }
            thread::sync_threads();
            if warp_id < 4 {
                let s = unsafe {
                    multiply::<256, 264, false>([0.0; 4], qhi, qlo, kv, warp_id * 8, lane)
                };
                let row = lane / 4;
                let col = warp_id * 8 + lane % 4 * 2;
                unsafe {
                    *scores.add((row * 32 + col) as usize) = s[0] / 16.0;
                    *scores.add((row * 32 + col + 1) as usize) = s[1] / 16.0;
                    *scores.add(((row + 8) * 32 + col) as usize) = s[2] / 16.0;
                    *scores.add(((row + 8) * 32 + col + 1) as usize) = s[3] / 16.0;
                }
            }
            thread::sync_threads();
            let mut r = 0;
            while r < 2 {
                let row = warp_id + r * 8;
                let valid = query_base + row < queries
                    && start + lane < tokens - batch + (query_base + row) / groups + 1
                    && start + lane < tokens;
                let score = if valid {
                    unsafe { *scores.add((row * 32 + lane) as usize) }
                } else {
                    f32::NEG_INFINITY
                };
                let mut maximum = score;
                let mut d = 16;
                while d > 0 {
                    maximum = maximum.max(warp::shuffle_down_f32_sync(0xffff_ffff, maximum, d));
                    d /= 2;
                }
                maximum = warp::shuffle_f32_sync(0xffff_ffff, maximum, 0).max(maxima[r as usize]);
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
                    let at = (row * 20 + lane / 2) as usize;
                    unsafe {
                        split(phi.add(at), plo.add(at), probability, next);
                    }
                }
                // Each output fragment owns rows lane/4 and lane/4+8.
                if lane == 0 {
                    unsafe {
                        *scores.add((512 + row) as usize) = factor;
                    }
                }
                r += 1;
            }
            thread::sync_threads();
            i = local;
            while i < 32 * 128 {
                let token = start + i / 128;
                let word = if token < tokens {
                    let at = ((token * kvh + kh) * 128 + i % 128) as usize;
                    unsafe { v.as_ptr().cast::<u32>().add(at).read() }
                } else {
                    0
                };
                unsafe {
                    kv.add((i / 128 * 132 + i % 128) as usize).write(word);
                }
                i += 256;
            }
            thread::sync_threads();
            let a = unsafe { *scores.add((512 + lane / 4) as usize) };
            let b = unsafe { *scores.add((520 + lane / 4) as usize) };
            let mut j = 0;
            #[unroll]
            while j < 4 {
                let old = output[j];
                output[j] = unsafe {
                    multiply::<32, 40, true>(
                        [old[0] * a, old[1] * a, old[2] * b, old[3] * b],
                        phi,
                        plo,
                        kv,
                        warp_id * 32 + j as u32 * 8,
                        lane,
                    )
                };
                j += 1;
            }
            thread::sync_threads();
            start += 32;
        }
        // Convert warp-owned row metadata into a shared table for all fragments.
        if lane == 0 {
            let mut r = 0;
            while r < 2 {
                unsafe {
                    *scores.add((512 + warp_id + r * 8) as usize) = maxima[r as usize];
                    *scores.add((528 + warp_id + r * 8) as usize) = totals[r as usize];
                }
                r += 1;
            }
        }
        thread::sync_threads();
        let mut j = 0;
        #[unroll]
        while j < 4 {
            let col = warp_id * 32 + j as u32 * 8 + lane % 4 * 2;
            let mut r = 0;
            while r < 2 {
                let row = lane / 4 + r * 8;
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
                        *partial.add(base + col as usize) = output[j][r as usize * 2];
                        *partial.add(base + col as usize + 1) = output[j][r as usize * 2 + 1];
                        if col == 0 {
                            *partial.add(base + 256) = *scores.add((512 + row) as usize);
                            *partial.add(base + 257) = *scores.add((528 + row) as usize);
                        }
                    }
                }
                r += 1;
            }
            j += 1;
        }
    }
}
