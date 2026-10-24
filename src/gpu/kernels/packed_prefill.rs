//! Direct packed-weight tiles for short prefill batches.
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    struct Words {
        data: [u32; 4],
    }

    #[inline(always)]
    fn pair(word: u32, shift: u32, scale: f32, offset: f32) -> u32 {
        let a = ((word >> shift) as u8 as i8 as f32) * scale + offset;
        let b = ((word >> (shift + 8)) as u8 as i8 as f32) * scale + offset;
        cuda_device::convert::cvt_f16x2_f32(a, b)
    }

    /// Decode a 64-row tile once and reuse it across up to 32 token columns.
    #[kernel]
    pub unsafe fn prefill_packed<const KIND: u32, const BYTES: usize>(
        cols: usize,
        rows: usize,
        batch: usize,
        weights: &[u8],
        input: &[u32],
        output: *mut f32,
    ) {
        static mut TILE: cuda_device::SharedArray<u32, 3456, 16> = cuda_device::SharedArray::UNINIT;
        let tid = thread::index_1d().get() as u32;
        let local = tid % 256;
        let lane = local % 32;
        let warp = local / 32;
        let cols = cols as u32;
        let rows = rows as u32;
        let batch = batch as u32;
        let tile_id = tid / 256;
        let row_base = tile_id % rows.div_ceil(64) * 64;
        let token_base = tile_id / rows.div_ceil(64) * 32;
        // SAFETY: every CTA fills disjoint words and reaches both barriers;
        // all row/token tails are padded before the warp matrix loads.
        let tile = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut TILE) };
        let mut sum0 = [0.0f32; 4];
        let mut sum1 = [0.0f32; 4];
        let mut col = 0;
        while col < cols {
            let mut vec = local;
            while vec < 512 {
                let row = row_base + vec / 8;
                let k = (vec % 8) * 8;
                let dest = ((vec / 8) * 36 + k / 2) as usize;
                if KIND == 1 {
                    let packed = if row < rows {
                        unsafe {
                            weights
                                .as_ptr()
                                .add(((row * cols + col + k) * 2) as usize)
                                .cast::<Words>()
                                .read()
                        }
                    } else {
                        Words { data: [0; 4] }
                    };
                    unsafe {
                        tile.add(dest).cast::<Words>().write(packed);
                    }
                } else if KIND == 0 || KIND == 30 {
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
            let mut i = local;
            while i < 1024 {
                let token = token_base + i / 32;
                let x = if token < batch {
                    unsafe {
                        *input.get_unchecked((token * (cols / 2) + col / 2 + i % 32) as usize)
                    }
                } else {
                    0
                };
                unsafe {
                    *tile.add((2304 + (i / 32) * 36 + i % 32) as usize) = x;
                }
                i += 256;
            }
            thread::sync_threads();
            let mut j = 0;
            while j < 64 {
                unsafe {
                    let a = cuda_device::wmma::ldmatrix_x4(tile.add(
                        (((warp / 2) * 16 + lane % 16) * 36 + (lane / 16) * 4 + j / 2) as usize,
                    ));
                    let b0 = cuda_device::wmma::ldmatrix_x2(tile.add(
                        (2304 + ((warp % 2) * 16 + lane % 8) * 36 + ((lane / 8) % 2) * 4 + j / 2)
                            as usize,
                    ));
                    let b1 = cuda_device::wmma::ldmatrix_x2(tile.add(
                        (2304
                            + ((warp % 2) * 16 + 8 + lane % 8) * 36
                            + ((lane / 8) % 2) * 4
                            + j / 2) as usize,
                    ));
                    sum0 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum0, a, b0);
                    sum1 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum1, a, b1);
                }
                j += 16;
            }
            thread::sync_threads();
            col += 64;
        }
        let row = row_base + (warp / 2) * 16 + lane / 4;
        let token = token_base + (warp % 2) * 16 + (lane % 4) * 2;
        // SAFETY: disjoint lane/warp outputs with explicit tail guards.
        unsafe {
            store(rows, batch, row, token, sum0, output);
            store(rows, batch, row, token + 8, sum1, output);
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
