//! Float32 matrix products using high/residual FP16 operands and FP32 sums.
use cuda_device::{convert, kernel, thread, wmma};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[inline(always)]
    pub(in crate::gpu::kernels) unsafe fn split(hi: *mut u32, lo: *mut u32, a: f32, b: f32) {
        let high = convert::cvt_f16x2_f32(a, b);
        let (x, y) = convert::cvt_f32x2_f16x2(high);
        unsafe {
            hi.write(high);
            lo.write(convert::cvt_f16x2_f32(a - x, b - y));
        }
    }
    #[inline(always)]
    pub(in crate::gpu::kernels) unsafe fn product(
        sum: [f32; 4],
        ah: [u32; 4],
        al: [u32; 4],
        bh: [u32; 2],
        bl: [u32; 2],
    ) -> [f32; 4] {
        // SAFETY: callers load complete matrix fragments in full converged warps.
        unsafe {
            let sum = wmma::mma_m16n8k16_f32_f16(sum, al, bh);
            let sum = wmma::mma_m16n8k16_f32_f16(sum, ah, bl);
            wmma::mma_m16n8k16_f32_f16(sum, ah, bh)
        }
    }
    #[kernel]
    pub unsafe fn float_matrix(
        cols: usize,
        rows: usize,
        batch: usize,
        weights: &[u8],
        input: &[f32],
        output: *mut f32,
    ) {
        static mut TILE: cuda_device::SharedArray<u32, 2560, 16> = cuda_device::SharedArray::UNINIT;
        let tile = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut TILE) };
        let tid = thread::index_1d().get();
        let local = tid % 128;
        let lane = local % 32;
        let warp = local / 32;
        let token_tiles = batch.div_ceil(32);
        let row_base = tid / 128 / token_tiles * 32;
        let token_base = tid / 128 % token_tiles * 32;
        let ah = tile;
        let al = unsafe { tile.add(640) };
        let bh = unsafe { tile.add(1280) };
        let bl = unsafe { tile.add(1920) };
        let mut sum0 = [0.0f32; 4];
        let mut sum1 = [0.0f32; 4];
        let mut col = 0;
        while col < cols {
            let mut i = local;
            while i < 512 {
                let row = row_base + i / 16;
                let k = col + i % 16 * 2;
                let at = i / 16 * 20 + i % 16;
                let (a, b) = if row < rows {
                    unsafe {
                        let p = weights.as_ptr().add((row * cols + k) * 4).cast::<f32>();
                        (p.read(), p.add(1).read())
                    }
                } else {
                    (0.0, 0.0)
                };
                let token = token_base + i / 16;
                let (x, y) = if token < batch {
                    (input[token * cols + k], input[token * cols + k + 1])
                } else {
                    (0.0, 0.0)
                };
                unsafe {
                    split(ah.add(at), al.add(at), a, b);
                    split(bh.add(at), bl.add(at), x, y);
                }
                i += 128;
            }
            thread::sync_threads();
            let mut k = 0;
            while k < 32 {
                let a = ((warp / 2) * 16 + lane % 16) * 20 + (lane / 16) * 4 + k / 2;
                let b = ((warp % 2) * 16 + lane % 8) * 20 + ((lane / 8) % 2) * 4 + k / 2;
                unsafe {
                    let high_a = wmma::ldmatrix_x4(ah.add(a));
                    let low_a = wmma::ldmatrix_x4(al.add(a));
                    sum0 = product(
                        sum0,
                        high_a,
                        low_a,
                        wmma::ldmatrix_x2(bh.add(b)),
                        wmma::ldmatrix_x2(bl.add(b)),
                    );
                    sum1 = product(
                        sum1,
                        high_a,
                        low_a,
                        wmma::ldmatrix_x2(bh.add(b + 160)),
                        wmma::ldmatrix_x2(bl.add(b + 160)),
                    );
                }
                k += 16;
            }
            thread::sync_threads();
            col += 32;
        }
        let row = row_base + (warp / 2) * 16 + lane / 4;
        let token = token_base + (warp % 2) * 16 + lane % 4 * 2;
        unsafe {
            store(rows, batch, row, token, sum0, output);
            store(rows, batch, row, token + 8, sum1, output);
        }
    }
    #[inline(always)]
    pub(in crate::gpu::kernels) unsafe fn store(
        rows: usize,
        batch: usize,
        row: usize,
        token: usize,
        sum: [f32; 4],
        output: *mut f32,
    ) {
        if row < rows && token < batch {
            unsafe {
                *output.add(token * rows + row) = sum[0];
            }
        }
        if row < rows && token + 1 < batch {
            unsafe {
                *output.add((token + 1) * rows + row) = sum[1];
            }
        }
        if row + 8 < rows && token < batch {
            unsafe {
                *output.add(token * rows + row + 8) = sum[2];
            }
        }
        if row + 8 < rows && token + 1 < batch {
            unsafe {
                *output.add((token + 1) * rows + row + 8) = sum[3];
            }
        }
    }
}
