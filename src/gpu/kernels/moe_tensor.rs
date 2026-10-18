//! Grouped expert tiles: original W2/BF16 scales, FP16 operands, FP32 sums.
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;

    // Preserve aligned eight-half groups while distributing matrix rows
    // across shared-memory banks, matching the independent GSQ tile layout.
    #[inline(always)]
    fn shared_index(row: usize, word: usize) -> usize {
        row * 32 + (word ^ ((row & 7) * 4))
    }

    #[inline(always)]
    fn value<const HOST: bool>(
        packed: &[u8],
        scales: &[u16],
        host: *const u8,
        host_scales: *const u16,
        columns: usize,
        row: usize,
        column: usize,
    ) -> f32 {
        let index = row * columns + column;
        let (byte, scale) = if !HOST || index / 4 < packed.len() {
            (packed[index / 4], scales[index / 128])
        } else {
            // SAFETY: the host launcher owns a mapped, immutable suffix with
            // exactly the remaining expert rows for this projection.
            unsafe {
                (
                    *host.add(index / 4 - packed.len()),
                    *host_scales.add(index / 128 - scales.len()),
                )
            }
        };
        let code = ((byte >> ((index % 4) * 2)) & 3) as i32 - 2;
        code as f32 * f32::from_bits((scale as u32) << 16)
    }

    #[kernel]
    pub unsafe fn matmul<const GATE: bool, const HOST: bool>(
        cols: usize,
        rows: usize,
        batch: usize,
        counts: &[u32],
        choices: &[u32],
        tiles: &[u32],
        weights: &[u8],
        scales: &[u16],
        up: &[u8],
        up_scales: &[u16],
        host_weights: *const u8,
        host_scales: *const u16,
        host_up: *const u8,
        host_up_scales: *const u16,
        input: &[f32],
        output: *mut f32,
    ) {
        static mut TILE: cuda_device::SharedArray<u32, 4608, 16> = cuda_device::SharedArray::UNINIT;
        let tid = thread::index_1d().get();
        let local = tid % 128;
        let lane = local % 32;
        let warp = local / 32;
        let block = tid / 128;
        let row_base = block % rows.div_ceil(64) * 64;
        let descriptor = block / rows.div_ceil(64);
        if descriptor >= tiles[0] as usize {
            return;
        }
        let expert = tiles[1 + descriptor * 2] as usize;
        let token_base = tiles[2 + descriptor * 2] as usize;
        // All live CTAs reach both barriers. The descriptor bounds guarantee
        // sixteen disjoint choices; row/choice tails are zero padded.
        let tile = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut TILE) };
        let mut sum0 = [0.0f32; 4];
        let mut sum1 = [0.0f32; 4];
        let mut upper0 = [0.0f32; 4];
        let mut upper1 = [0.0f32; 4];
        let mut col = 0;
        while col < cols {
            let mut pair = local;
            while pair < 2048 {
                let row = row_base + pair / 32;
                let column = col + pair % 32 * 2;
                let (a, b, u, v) = if row < rows {
                    let r = expert * rows + row;
                    (
                        value::<HOST>(weights, scales, host_weights, host_scales, cols, r, column),
                        value::<HOST>(
                            weights,
                            scales,
                            host_weights,
                            host_scales,
                            cols,
                            r,
                            column + 1,
                        ),
                        if GATE {
                            value::<HOST>(up, up_scales, host_up, host_up_scales, cols, r, column)
                        } else {
                            0.0
                        },
                        if GATE {
                            value::<HOST>(
                                up,
                                up_scales,
                                host_up,
                                host_up_scales,
                                cols,
                                r,
                                column + 1,
                            )
                        } else {
                            0.0
                        },
                    )
                } else {
                    (0.0, 0.0, 0.0, 0.0)
                };
                unsafe {
                    *tile.add(shared_index(pair / 32, pair % 32)) =
                        cuda_device::convert::cvt_f16x2_f32(a, b);
                    if GATE {
                        *tile.add(2048 + shared_index(pair / 32, pair % 32)) =
                            cuda_device::convert::cvt_f16x2_f32(u, v);
                    }
                }
                pair += 128;
            }
            let mut pair = local;
            while pair < 512 {
                let slot = token_base + pair / 32;
                let column = col + pair % 32 * 2;
                let (a, b) = if slot < counts[expert] as usize {
                    let choice = choices[expert * batch + slot] as usize;
                    let source = if GATE { choice / 8 } else { choice };
                    let at = source * cols + column;
                    (input[at], input[at + 1])
                } else {
                    (0.0, 0.0)
                };
                unsafe {
                    *tile.add(4096 + shared_index(pair / 32, pair % 32)) =
                        cuda_device::convert::cvt_f16x2_f32(a, b);
                }
                pair += 128;
            }
            thread::sync_threads();
            let mut k = 0;
            while k < 64 {
                unsafe {
                    let offset = shared_index(warp * 16 + lane % 16, lane / 16 * 4 + k / 2);
                    let a = cuda_device::wmma::ldmatrix_x4(tile.add(offset));
                    let b0 = cuda_device::wmma::ldmatrix_x2(
                        tile.add(4096 + shared_index(lane % 8, lane / 8 % 2 * 4 + k / 2)),
                    );
                    let b1 = cuda_device::wmma::ldmatrix_x2(
                        tile.add(4096 + shared_index(8 + lane % 8, lane / 8 % 2 * 4 + k / 2)),
                    );
                    sum0 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum0, a, b0);
                    sum1 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum1, a, b1);
                    if GATE {
                        let a = cuda_device::wmma::ldmatrix_x4(tile.add(2048 + offset));
                        upper0 = cuda_device::wmma::mma_m16n8k16_f32_f16(upper0, a, b0);
                        upper1 = cuda_device::wmma::mma_m16n8k16_f32_f16(upper1, a, b1);
                    }
                }
                k += 16;
            }
            thread::sync_threads();
            col += 64;
        }
        let row = row_base + warp * 16 + lane / 4;
        let token = token_base + lane % 4 * 2;
        unsafe {
            store::<GATE>(
                rows, batch, expert, row, token, sum0, upper0, counts, choices, output,
            );
            store::<GATE>(
                rows,
                batch,
                expert,
                row,
                token + 8,
                sum1,
                upper1,
                counts,
                choices,
                output,
            );
        }
    }

    #[inline(always)]
    unsafe fn store<const GATE: bool>(
        rows: usize,
        batch: usize,
        expert: usize,
        row: usize,
        token: usize,
        sums: [f32; 4],
        upper: [f32; 4],
        counts: &[u32],
        choices: &[u32],
        output: *mut f32,
    ) {
        let mut i = 0;
        while i < 4 {
            let r = row + if i >= 2 { 8 } else { 0 };
            let slot = token + i % 2;
            if r < rows && slot < counts[expert] as usize {
                let choice = choices[expert * batch + slot] as usize;
                let value = if GATE {
                    sums[i] / (1.0 + (-sums[i]).exp()) * upper[i]
                } else {
                    sums[i]
                };
                unsafe {
                    *output.add(choice * rows + r) = value;
                }
            }
            i += 1;
        }
    }
}
