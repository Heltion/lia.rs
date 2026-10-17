//! Packed weight tiles reused across wide token groups without global expansion.
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    struct Words4 {
        data: [u32; 4],
    }

    #[inline(always)]
    fn pair(word: u32, shift: u32, scale: f32, offset: f32) -> u32 {
        let a = ((word >> shift) as u8 as i8 as f32) * scale + offset;
        let b = ((word >> (shift + 8)) as u8 as i8 as f32) * scale + offset;
        cuda_device::convert::cvt_f16x2_f32(a, b)
    }

    /// Reuse each decoded 64-row weight tile across 128 or 256 token columns.
    #[kernel]
    pub unsafe fn prefill_wide<const KIND: u32, const BYTES: usize, const TOKENS: usize>(
        cols: usize,
        rows: usize,
        batch: usize,
        weights: &[u8],
        input: &[u32],
        output: *mut f32,
    ) {
        static mut TILE: cuda_device::SharedArray<u32, 11520, 16> =
            cuda_device::SharedArray::UNINIT;
        let tid = thread::index_1d().get() as u32;
        let local = tid % 256;
        let lane = local % 32;
        let warp = local / 32;
        let cols = cols as u32;
        let rows = rows as u32;
        let batch = batch as u32;
        let tile_id = tid / 256;
        let row_base = tile_id % rows.div_ceil(64) * 64;
        let token_base = tile_id / rows.div_ceil(64) * TOKENS as u32;
        // SAFETY: every CTA fills disjoint words and reaches both barriers;
        // all row/token tails are padded before the warp matrix loads.
        let tile = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut TILE) };
        let mut sum00 = [0.0f32; 4];
        let mut sum01 = [0.0f32; 4];
        let mut sum02 = [0.0f32; 4];
        let mut sum03 = [0.0f32; 4];
        let mut sum04 = [0.0f32; 4];
        let mut sum05 = [0.0f32; 4];
        let mut sum06 = [0.0f32; 4];
        let mut sum07 = [0.0f32; 4];
        let mut sum10 = [0.0f32; 4];
        let mut sum11 = [0.0f32; 4];
        let mut sum12 = [0.0f32; 4];
        let mut sum13 = [0.0f32; 4];
        let mut sum14 = [0.0f32; 4];
        let mut sum15 = [0.0f32; 4];
        let mut sum16 = [0.0f32; 4];
        let mut sum17 = [0.0f32; 4];
        let mut col = 0;
        while col < cols {
            let mut vec = local;
            while vec < 512 {
                let row = row_base + vec / 8;
                let k = (vec % 8) * 8;
                let dest = ((vec / 8) * 36 + k / 2) as usize;
                if KIND == 0 || KIND == 30 {
                    let mut j = 0;
                    while j < 4 {
                        let (a, b) = if row < rows {
                            let index = (row * cols + col + k + j * 2) as usize;
                            (
                                crate::quant::decode_value(
                                    KIND,
                                    &weights[index * BYTES..(index + 1) * BYTES],
                                    0,
                                ),
                                crate::quant::decode_value(
                                    KIND,
                                    &weights[(index + 1) * BYTES..(index + 2) * BYTES],
                                    0,
                                ),
                            )
                        } else {
                            (0.0, 0.0)
                        };
                        unsafe {
                            *tile.add(dest + j as usize) =
                                cuda_device::convert::cvt_f16x2_f32(a, b);
                        }
                        j += 1;
                    }
                } else {
                    let (lo, hi, scale, offset) = if row < rows {
                        let block = (row * cols + col + k) / 256;
                        let data = unsafe {
                            core::slice::from_raw_parts(
                                weights.as_ptr().add(block as usize * BYTES),
                                BYTES,
                            )
                        };
                        unsafe {
                            crate::quant::packed_eight_aligned(
                                KIND,
                                data,
                                ((col + k) % 256) as usize,
                            )
                        }
                    } else {
                        (0, 0, 0.0, 0.0)
                    };
                    unsafe {
                        *tile.add(dest) = pair(lo, 0, scale, offset);
                        *tile.add(dest + 1) = pair(lo, 16, scale, offset);
                        *tile.add(dest + 2) = pair(hi, 0, scale, offset);
                        *tile.add(dest + 3) = pair(hi, 16, scale, offset);
                    }
                }
                vec += 256;
            }
            let mut i = local * 4;
            while i < TOKENS as u32 * 32 {
                let token = token_base + i / 32;
                let x = if token < batch {
                    unsafe {
                        input
                            .as_ptr()
                            .add((token * (cols / 2) + col / 2 + i % 32) as usize)
                            .cast::<Words4>()
                            .read()
                    }
                } else {
                    Words4 { data: [0; 4] }
                };
                unsafe {
                    tile.add((2304 + (i / 32) * 36 + i % 32) as usize)
                        .cast::<Words4>()
                        .write(x);
                }
                i += 1024;
            }
            thread::sync_threads();
            let group = (warp / 4) * 32;
            let token_group = (warp % 4) * (TOKENS as u32 / 4);
            let mut j = 0;
            while j < 64 {
                // SAFETY: completed padded tiles match each MMA fragment layout.
                unsafe {
                    let a0 = cuda_device::wmma::ldmatrix_x4(
                        tile.add(((group + lane % 16) * 36 + (lane / 16) * 4 + j / 2) as usize),
                    );
                    let a1 =
                        cuda_device::wmma::ldmatrix_x4(tile.add(
                            ((group + 16 + lane % 16) * 36 + (lane / 16) * 4 + j / 2) as usize,
                        ));
                    if TOKENS >= 32 {
                        let b = cuda_device::wmma::ldmatrix_x2(tile.add(
                            (2304 + (token_group + lane % 8) * 36 + ((lane / 8) % 2) * 4 + j / 2)
                                as usize,
                        ));
                        sum00 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum00, a0, b);
                        sum10 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum10, a1, b);
                    }
                    if TOKENS >= 64 {
                        let b = cuda_device::wmma::ldmatrix_x2(tile.add(
                            (2304
                                + (token_group + 8 + lane % 8) * 36
                                + ((lane / 8) % 2) * 4
                                + j / 2) as usize,
                        ));
                        sum01 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum01, a0, b);
                        sum11 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum11, a1, b);
                    }
                    if TOKENS >= 96 {
                        let b = cuda_device::wmma::ldmatrix_x2(tile.add(
                            (2304
                                + (token_group + 16 + lane % 8) * 36
                                + ((lane / 8) % 2) * 4
                                + j / 2) as usize,
                        ));
                        sum02 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum02, a0, b);
                        sum12 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum12, a1, b);
                    }
                    if TOKENS >= 128 {
                        let b = cuda_device::wmma::ldmatrix_x2(tile.add(
                            (2304
                                + (token_group + 24 + lane % 8) * 36
                                + ((lane / 8) % 2) * 4
                                + j / 2) as usize,
                        ));
                        sum03 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum03, a0, b);
                        sum13 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum13, a1, b);
                    }
                    if TOKENS >= 160 {
                        let b = cuda_device::wmma::ldmatrix_x2(tile.add(
                            (2304
                                + (token_group + 32 + lane % 8) * 36
                                + ((lane / 8) % 2) * 4
                                + j / 2) as usize,
                        ));
                        sum04 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum04, a0, b);
                        sum14 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum14, a1, b);
                    }
                    if TOKENS >= 192 {
                        let b = cuda_device::wmma::ldmatrix_x2(tile.add(
                            (2304
                                + (token_group + 40 + lane % 8) * 36
                                + ((lane / 8) % 2) * 4
                                + j / 2) as usize,
                        ));
                        sum05 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum05, a0, b);
                        sum15 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum15, a1, b);
                    }
                    if TOKENS >= 224 {
                        let b = cuda_device::wmma::ldmatrix_x2(tile.add(
                            (2304
                                + (token_group + 48 + lane % 8) * 36
                                + ((lane / 8) % 2) * 4
                                + j / 2) as usize,
                        ));
                        sum06 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum06, a0, b);
                        sum16 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum16, a1, b);
                    }
                    if TOKENS >= 256 {
                        let b = cuda_device::wmma::ldmatrix_x2(tile.add(
                            (2304
                                + (token_group + 56 + lane % 8) * 36
                                + ((lane / 8) % 2) * 4
                                + j / 2) as usize,
                        ));
                        sum07 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum07, a0, b);
                        sum17 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum17, a1, b);
                    }
                }
                j += 16;
            }
            thread::sync_threads();
            col += 64;
        }
        let row = row_base + (warp / 4) * 32 + lane / 4;
        let token = token_base + (warp % 4) * (TOKENS as u32 / 4) + (lane % 4) * 2;
        // SAFETY: disjoint lane/warp outputs with explicit tail guards.
        unsafe {
            if TOKENS >= 32 {
                store(rows, batch, row, token, sum00, output);
                store(rows, batch, row + 16, token, sum10, output);
            }
            if TOKENS >= 64 {
                store(rows, batch, row, token + 8, sum01, output);
                store(rows, batch, row + 16, token + 8, sum11, output);
            }
            if TOKENS >= 96 {
                store(rows, batch, row, token + 16, sum02, output);
                store(rows, batch, row + 16, token + 16, sum12, output);
            }
            if TOKENS >= 128 {
                store(rows, batch, row, token + 24, sum03, output);
                store(rows, batch, row + 16, token + 24, sum13, output);
            }
            if TOKENS >= 160 {
                store(rows, batch, row, token + 32, sum04, output);
                store(rows, batch, row + 16, token + 32, sum14, output);
            }
            if TOKENS >= 192 {
                store(rows, batch, row, token + 40, sum05, output);
                store(rows, batch, row + 16, token + 40, sum15, output);
            }
            if TOKENS >= 224 {
                store(rows, batch, row, token + 48, sum06, output);
                store(rows, batch, row + 16, token + 48, sum16, output);
            }
            if TOKENS >= 256 {
                store(rows, batch, row, token + 56, sum07, output);
                store(rows, batch, row + 16, token + 56, sum17, output);
            }
        }
    }

    #[inline(always)]
    unsafe fn store(rows: u32, batch: u32, row: u32, token: u32, sums: [f32; 4], output: *mut f32) {
        unsafe {
            if row < rows && token < batch {
                *output.add((token * rows + row) as usize) = sums[0];
            }
            if row < rows && token + 1 < batch {
                *output.add(((token + 1) * rows + row) as usize) = sums[1];
            }
            if row + 8 < rows && token < batch {
                *output.add((token * rows + row + 8) as usize) = sums[2];
            }
            if row + 8 < rows && token + 1 < batch {
                *output.add(((token + 1) * rows + row + 8) as usize) = sums[3];
            }
        }
    }
}
