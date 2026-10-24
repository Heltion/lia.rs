//! Decode packed AWQ tiles once for32 tokens, retaining FP32 high/residual products.
use super::float_matrix::device::{product, split, store};
use cuda_device::{convert, kernel, thread, wmma};
use cuda_host::cuda_module;
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub unsafe fn awq_matrix<const RESIDUAL: bool>(
        cols: usize,
        rows: usize,
        group_size: usize,
        batch: usize,
        weights: &[u32],
        zeros: &[u32],
        scales: &[f32],
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
        let groups = cols / group_size;
        let mut group = 0;
        let mut next_group = group_size;
        while col < cols {
            // One32-column tile stays within one complete quantization group.
            if col == next_group {
                group += 1;
                next_group += group_size;
            }
            let mut i = local;
            while i < 512 {
                // Each warp spans four alternating rows and eight pairs.
                // Stride20 then maps its32 stores to distinct shared banks.
                let tile_row = i / 128 * 8 + (i % 32 / 8) * 2 + (i / 32 % 2);
                let pair = i % 8 + (i / 64 % 2) * 8;
                let row = row_base + tile_row;
                let k = col + pair * 2;
                let at = tile_row * 20 + pair;
                let (a, b) = if row < rows {
                    let nibble = row % 8;
                    let shift = ((nibble % 2) * 4 + nibble / 2) * 4;
                    let zero = ((zeros[row / 8 * groups + group] >> shift) & 15) as i32;
                    let gain = scales[group * rows + row];
                    // Device allocation and even k give an aligned8-byte load.
                    let packed = unsafe {
                        weights
                            .as_ptr()
                            .add(row / 8 * cols + k)
                            .cast::<u64>()
                            .read()
                    };
                    let first = (((packed as u32) >> shift) & 15) as i32;
                    let second = ((((packed >> 32) as u32) >> shift) & 15) as i32;
                    ((first - zero) as f32 * gain, (second - zero) as f32 * gain)
                } else {
                    (0., 0.)
                };
                let token = token_base + tile_row;
                let (x, y) = if token < batch {
                    (input[token * cols + k], input[token * cols + k + 1])
                } else {
                    (0., 0.)
                };
                // Complete padded32x32 tiles; group boundaries are32-aligned.
                unsafe {
                    if RESIDUAL {
                        split(ah.add(at), al.add(at), a, b);
                        split(bh.add(at), bl.add(at), x, y);
                    } else {
                        ah.add(at).write(convert::cvt_f16x2_f32(a, b));
                        bh.add(at).write(convert::cvt_f16x2_f32(x, y));
                    }
                }
                i += 128;
            }
            thread::sync_threads();
            let mut k = 0;
            while k < 32 {
                let a = ((warp / 2) * 16 + lane % 16) * 20 + (lane / 16) * 4 + k / 2;
                let b = ((warp % 2) * 16 + lane % 8) * 20 + ((lane / 8) % 2) * 4 + k / 2;
                // Complete converged warps load the same padded operand tiles.
                unsafe {
                    let high_a = wmma::ldmatrix_x4(ah.add(a));
                    let high_b0 = wmma::ldmatrix_x2(bh.add(b));
                    let high_b1 = wmma::ldmatrix_x2(bh.add(b + 160));
                    if RESIDUAL {
                        let low_a = wmma::ldmatrix_x4(al.add(a));
                        sum0 = product(sum0, high_a, low_a, high_b0, wmma::ldmatrix_x2(bl.add(b)));
                        sum1 = product(
                            sum1,
                            high_a,
                            low_a,
                            high_b1,
                            wmma::ldmatrix_x2(bl.add(b + 160)),
                        );
                    } else {
                        sum0 = wmma::mma_m16n8k16_f32_f16(sum0, high_a, high_b0);
                        sum1 = wmma::mma_m16n8k16_f32_f16(sum1, high_a, high_b1);
                    }
                }
                k += 16;
            }
            thread::sync_threads();
            col += 32;
        }
        let row = row_base + (warp / 2) * 16 + lane / 4;
        let token = token_base + (warp % 2) * 16 + lane % 4 * 2;
        // Disjoint warp/lane output elements with guarded row and token tails.
        unsafe {
            store(rows, batch, row, token, sum0, output);
            store(rows, batch, row, token + 8, sum1, output);
        }
    }
}
