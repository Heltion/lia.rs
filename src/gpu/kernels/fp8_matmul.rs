//! Quantized weights decoded directly into FP8 tiles with persistent F32 accumulation.
use cuda_device::{kernel, launch_bounds, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;
    use crate::gpu::kernels::fp8_weights::packed;
    const TOKENS: u32 = 128;

    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    struct Words4 {
        data: [u32; 4],
    }

    #[inline(always)]
    fn shared_index(row: u32, word: u32) -> usize {
        (row * 20 + word) as usize
    }

    #[kernel]
    #[launch_bounds(256, 4)]
    pub unsafe fn prefill_fp8<const KIND: u32, const BYTES: usize, const CACHED: bool>(
        cols: usize,
        rows: usize,
        batch: usize,
        weight_scale: f32,
        scales: &[f32],
        weights: &[u8],
        input: &[u32],
        cached: &[u32],
        output: *mut f32,
    ) {
        let tid = thread::index_1d().get() as u32;
        let local = tid % 256;
        let lane = local % 32;
        let warp = local / 32;
        let cols = cols as u32;
        let rows = rows as u32;
        let batch = batch as u32;
        let tile_id = tid / 256;
        // Adjacent CTAs consume the same weight rows for neighboring token tiles,
        // retaining their working set in L2 instead of sweeping the full matrix.
        let token_tiles = batch.div_ceil(TOKENS);
        let row_base = tile_id / token_tiles * 64;
        let token_base = tile_id % token_tiles * TOKENS;
        // SAFETY: every CTA fills disjoint words and reaches both barriers;
        // all row/token tails are padded before the warp matrix loads.
        static mut TILE: cuda_device::SharedArray<u32, 3840, 16> = cuda_device::SharedArray::UNINIT;
        let tile = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut TILE) };
        let weight_inverse = 1.0 / weight_scale;
        let input_inverse = scales[1];
        let mut sum00 = [0.0f32; 4];
        let mut sum01 = [0.0f32; 4];
        let mut sum02 = [0.0f32; 4];
        let mut sum03 = [0.0f32; 4];
        let mut sum10 = [0.0f32; 4];
        let mut sum11 = [0.0f32; 4];
        let mut sum12 = [0.0f32; 4];
        let mut sum13 = [0.0f32; 4];

        let mut col = 0;
        while col < cols {
            if CACHED {
                let local_row = local / 4;
                let word = (local % 4) * 4;
                let at = (row_base * cols + col * 64) / 4 + local_row * 16 + word;
                // SAFETY: load-time packing pads rows to64 and each vector is aligned.
                unsafe {
                    let x = cached.as_ptr().add(at as usize).cast::<Words4>().read();
                    tile.add(shared_index(local_row, word))
                        .cast::<Words4>()
                        .write(x);
                }
            } else {
                let mut vec = local;
                while vec < 512 {
                    let row = row_base + vec / 8;
                    let k = (vec % 8) * 8;
                    let dest = shared_index(vec / 8, k / 4);
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
                        *tile.add(dest) = packed(lo, scale, offset, weight_scale);
                        *tile.add(dest + 1) = packed(hi, scale, offset, weight_scale);
                    }
                    vec += 256;
                }
            }
            let mut i = local * 4;
            while i < TOKENS * 16 {
                let token = token_base + i / 16;
                let x = if token < batch {
                    unsafe {
                        input
                            .as_ptr()
                            .add((token * (cols / 4) + col / 4 + i % 16) as usize)
                            .cast::<Words4>()
                            .read()
                    }
                } else {
                    Words4 { data: [0; 4] }
                };
                unsafe {
                    tile.add(1280 + shared_index(i / 16, i % 16))
                        .cast::<Words4>()
                        .write(x);
                }
                i += 1024;
            }
            thread::sync_threads();
            let group = (warp / 4) * 32;
            let token_group = (warp % 4) * (TOKENS / 4);
            let mut j = 0usize;
            #[unroll]
            while j < 2 {
                // SAFETY: completed padded tiles match each MMA fragment layout.
                unsafe {
                    let a0 = cuda_device::wmma::ldmatrix_x4(tile.add(shared_index(
                        group + lane % 16,
                        (lane / 16) * 4 + j as u32 * 8,
                    )));
                    let a1 = cuda_device::wmma::ldmatrix_x4(tile.add(shared_index(
                        group + 16 + lane % 16,
                        (lane / 16) * 4 + j as u32 * 8,
                    )));

                    macro_rules! product {
                        ($offset:literal, $upper:ident, $lower:ident) => {{
                            let at = 1280
                                + shared_index(
                                    token_group + $offset + lane / 4,
                                    j as u32 * 8 + lane % 4,
                                );
                            let b = [*tile.add(at), *tile.add(at + 4)];
                            $upper =
                                cuda_device::wmma::mma_m16n8k32_fp8_f32_e4m3_e4m3($upper, a0, b);
                            $lower =
                                cuda_device::wmma::mma_m16n8k32_fp8_f32_e4m3_e4m3($lower, a1, b);
                        }};
                    }
                    product!(0, sum00, sum10);
                    product!(8, sum01, sum11);
                    if TOKENS >= 128 {
                        product!(16, sum02, sum12);
                        product!(24, sum03, sum13);
                    }
                }
                j += 1;
            }
            thread::sync_threads();
            col += 64;
        }
        let row = row_base + (warp / 4) * 32 + lane / 4;
        let token = token_base + (warp % 4) * (TOKENS / 4) + (lane % 4) * 2;
        macro_rules! emit {
            ($r:expr, $t:expr, $sum:ident) => {
                store(
                    rows,
                    batch,
                    $r,
                    $t,
                    $sum,
                    weight_inverse,
                    input_inverse,
                    output,
                );
            };
        }
        // SAFETY: unique lane/warp outputs, with explicit row/token tail guards.
        unsafe {
            emit!(row, token, sum00);
            emit!(row + 16, token, sum10);
            emit!(row, token + 8, sum01);
            emit!(row + 16, token + 8, sum11);
            emit!(row, token + 16, sum02);
            emit!(row + 16, token + 16, sum12);
            emit!(row, token + 24, sum03);
            emit!(row + 16, token + 24, sum13);
        }
    }

    #[inline(always)]
    unsafe fn store(
        rows: u32,
        batch: u32,
        row: u32,
        token: u32,
        sums: [f32; 4],
        weight_inverse: f32,
        input_inverse: f32,
        output: *mut f32,
    ) {
        unsafe {
            if row < rows && token < batch {
                *output.add((token * rows + row) as usize) =
                    sums[0] * weight_inverse * input_inverse;
            }
            if row < rows && token + 1 < batch {
                *output.add(((token + 1) * rows + row) as usize) =
                    sums[1] * weight_inverse * input_inverse;
            }
            if row + 8 < rows && token < batch {
                *output.add((token * rows + row + 8) as usize) =
                    sums[2] * weight_inverse * input_inverse;
            }
            if row + 8 < rows && token + 1 < batch {
                *output.add(((token + 1) * rows + row + 8) as usize) =
                    sums[3] * weight_inverse * input_inverse;
            }
        }
    }
}
