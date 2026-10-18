//! Noncausal shared-query tiles, width72, split FP16 operands and FP32 sums.
use cuda_device::{convert, kernel, thread, warp, wmma};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    struct Float4 {
        values: [f32; 4],
    }

    #[inline(always)]
    fn load4(
        qkv: &[f32],
        tokens: usize,
        token: usize,
        head: usize,
        column: usize,
        section: usize,
    ) -> [f32; 4] {
        if token < tokens && column < 72 {
            // SAFETY: complete width-72 heads and four-column offsets preserve
            // 16-byte alignment, including final groups and padded token tails.
            unsafe {
                qkv.as_ptr()
                    .add(token * 3456 + section + head * 72 + column)
                    .cast::<Float4>()
                    .read()
                    .values
            }
        } else {
            [0.0; 4]
        }
    }

    #[inline(always)]
    fn exp(x: f32) -> f32 {
        cuda_device::float::ex2_approx_f32(x * core::f32::consts::LOG2_E)
    }

    #[inline(always)]
    unsafe fn split(hi: *mut u32, lo: *mut u32, a: f32, b: f32) {
        let value = convert::cvt_f16x2_f32(a, b);
        let (ah, bh) = convert::cvt_f32x2_f16x2(value);
        unsafe {
            hi.write(value);
            lo.write(convert::cvt_f16x2_f32(a - ah, b - bh));
        }
    }

    #[inline(always)]
    unsafe fn multiply<const K: usize, const STRIDE: usize, const TRANS: bool>(
        mut sum: [f32; 4],
        a: *const u32,
        al: *const u32,
        b: *const u32,
        bl: *const u32,
        column: usize,
        lane: usize,
    ) -> [f32; 4] {
        let mut k = 0;
        while k < K {
            let offset = (lane % 16 * STRIDE + k + lane / 16 * 8) / 2;
            let bo = if TRANS {
                (k + lane % 16) * 44 + column / 2
            } else {
                (column + lane % 8) * 44 + k / 2 + lane / 8 % 2 * 4
            };
            unsafe {
                let ah = wmma::ldmatrix_x4(a.add(offset));
                let low = wmma::ldmatrix_x4(al.add(offset));
                let bh = if TRANS {
                    wmma::ldmatrix_x2_trans(b.add(bo))
                } else {
                    wmma::ldmatrix_x2(b.add(bo))
                };
                let low_b = if TRANS {
                    wmma::ldmatrix_x2_trans(bl.add(bo))
                } else {
                    wmma::ldmatrix_x2(bl.add(bo))
                };
                sum = wmma::mma_m16n8k16_f32_f16(sum, low, bh);
                sum = wmma::mma_m16n8k16_f32_f16(sum, ah, low_b);
                sum = wmma::mma_m16n8k16_f32_f16(sum, ah, bh);
            }
            k += 16;
        }
        sum
    }

    #[kernel]
    pub unsafe fn tiled_attention<const QUERIES: usize>(
        tokens: usize,
        qkv: &[f32],
        output: *mut f32,
    ) {
        let tile = if QUERIES == 32 {
            static mut TILE32: cuda_device::SharedArray<u32, 7968, 16> =
                cuda_device::SharedArray::UNINIT;
            unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut TILE32) }
        } else {
            static mut TILE16: cuda_device::SharedArray<u32, 5392, 16> =
                cuda_device::SharedArray::UNINIT;
            unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut TILE16) }
        };
        let tid = thread::index_1d().get();
        let local = tid % (QUERIES * 8);
        let lane = local % 32;
        let warp_group = local / 128;
        let warp_id = local / 32 % 4;
        let head = tid / (QUERIES * 8) % 16;
        let query_base = tid / (QUERIES * 8) / 16 * QUERIES;
        let qhi = tile;
        let qlo = unsafe { tile.add(QUERIES * 44) };
        let kv = unsafe { tile.add(QUERIES * 88) };
        let kvlo = unsafe { tile.add(QUERIES * 88 + 1408) };
        let phi = unsafe { tile.add(QUERIES * 88 + 2816) };
        let plo = unsafe { tile.add(QUERIES * 108 + 2816) };
        let scores = unsafe { tile.add(QUERIES * 128 + 2816).cast::<f32>() };
        let mut maxima = [f32::NEG_INFINITY; 4];
        let mut totals = [0.0f32; 4];
        let mut result = [[0.0f32; 4]; 3];
        let mut i = local;
        while i < QUERIES * 20 {
            let query = query_base + i / 20;
            let x = load4(qkv, tokens, query, head, i % 20 * 4, 0);
            let at = i / 20 * 44 + i % 20 * 2;
            unsafe {
                split(qhi.add(at), qlo.add(at), x[0], x[1]);
                split(qhi.add(at + 1), qlo.add(at + 1), x[2], x[3]);
            }
            i += QUERIES * 8;
        }
        thread::sync_threads();
        let mut start = 0;
        while start < tokens {
            i = local;
            while i < 32 * 20 {
                let token = start + i / 20;
                let x = load4(qkv, tokens, token, head, i % 20 * 4, 1152);
                let at = i / 20 * 44 + i % 20 * 2;
                unsafe {
                    split(kv.add(at), kvlo.add(at), x[0], x[1]);
                    split(kv.add(at + 1), kvlo.add(at + 1), x[2], x[3]);
                }
                i += QUERIES * 8;
            }
            thread::sync_threads();
            let sums = unsafe {
                multiply::<80, 88, false>(
                    [0.0; 4],
                    qhi.add(warp_group * 704),
                    qlo.add(warp_group * 704),
                    kv,
                    kvlo,
                    warp_id * 8,
                    lane,
                )
            };
            let row = warp_group * 16 + lane / 4;
            let column = warp_id * 8 + lane % 4 * 2;
            unsafe {
                *scores.add(row * 32 + column) = sums[0] / 72.0f32.sqrt();
                *scores.add(row * 32 + column + 1) = sums[1] / 72.0f32.sqrt();
                *scores.add((row + 8) * 32 + column) = sums[2] / 72.0f32.sqrt();
                *scores.add((row + 8) * 32 + column + 1) = sums[3] / 72.0f32.sqrt();
            }
            thread::sync_threads();
            let mut r = 0;
            while r < 4 {
                let row = warp_group * 16 + warp_id + r * 4;
                let valid = query_base + row < tokens && start + lane < tokens;
                let score = if valid {
                    unsafe { *scores.add(row * 32 + lane) }
                } else {
                    f32::NEG_INFINITY
                };
                let mut maximum = score;
                let mut d = 16;
                while d > 0 {
                    maximum = maximum.max(warp::shuffle_down_f32_sync(0xffff_ffff, maximum, d));
                    d /= 2;
                }
                maximum = warp::shuffle_f32_sync(0xffff_ffff, maximum, 0).max(maxima[r]);
                let factor = if maxima[r] == f32::NEG_INFINITY {
                    0.0
                } else {
                    exp(maxima[r] - maximum)
                };
                let probability = if valid { exp(score - maximum) } else { 0.0 };
                let mut total = probability;
                d = 16;
                while d > 0 {
                    total += warp::shuffle_down_f32_sync(0xffff_ffff, total, d);
                    d /= 2;
                }
                totals[r] = totals[r] * factor + warp::shuffle_f32_sync(0xffff_ffff, total, 0);
                maxima[r] = maximum;
                let next = warp::shuffle_down_f32_sync(0xffff_ffff, probability, 1);
                if lane.is_multiple_of(2) {
                    let at = row * 20 + lane / 2;
                    unsafe {
                        split(phi.add(at), plo.add(at), probability, next);
                    }
                }
                if lane == 0 {
                    unsafe {
                        *scores.add(QUERIES * 32 + row) = factor;
                    }
                }
                r += 1;
            }
            thread::sync_threads();
            i = local;
            while i < 32 * 20 {
                let token = start + i / 20;
                let x = load4(qkv, tokens, token, head, i % 20 * 4, 2304);
                let at = i / 20 * 44 + i % 20 * 2;
                unsafe {
                    split(kv.add(at), kvlo.add(at), x[0], x[1]);
                    split(kv.add(at + 1), kvlo.add(at + 1), x[2], x[3]);
                }
                i += QUERIES * 8;
            }
            thread::sync_threads();
            let a = unsafe { *scores.add(QUERIES * 32 + warp_group * 16 + lane / 4) };
            let b = unsafe { *scores.add(QUERIES * 32 + warp_group * 16 + 8 + lane / 4) };
            let mut j = 0;
            while j < 3 {
                let column = warp_id * 24 + j * 8;
                if column < 72 {
                    let old = result[j];
                    result[j] = unsafe {
                        multiply::<32, 40, true>(
                            [old[0] * a, old[1] * a, old[2] * b, old[3] * b],
                            phi.add(warp_group * 320),
                            plo.add(warp_group * 320),
                            kv,
                            kvlo,
                            column,
                            lane,
                        )
                    };
                }
                j += 1;
            }
            thread::sync_threads();
            start += 32;
        }
        if lane == 0 {
            let mut r = 0;
            while r < 4 {
                unsafe {
                    *scores.add(QUERIES * 32 + warp_group * 16 + warp_id + r * 4) = totals[r];
                }
                r += 1;
            }
        }
        thread::sync_threads();
        let mut j = 0;
        while j < 3 {
            let column = warp_id * 24 + j * 8 + lane % 4 * 2;
            let mut r = 0;
            while r < 2 {
                let row = warp_group * 16 + lane / 4 + r * 8;
                let query = query_base + row;
                if query < tokens && column < 72 {
                    let at = query * 1152 + head * 72 + column;
                    let total = unsafe { *scores.add(QUERIES * 32 + row) };
                    unsafe {
                        *output.add(at) = result[j][r * 2] / total;
                        *output.add(at + 1) = result[j][r * 2 + 1] / total;
                    }
                }
                r += 1;
            }
            j += 1;
        }
    }
}
