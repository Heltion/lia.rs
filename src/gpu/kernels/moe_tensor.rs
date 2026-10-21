//! Grouped expert tiles: original W2/W4 with BF16 scales, FP16 operands, FP32 sums.
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    struct Float4 {
        values: [f32; 4],
    }

    // Preserve aligned eight-half groups while distributing matrix rows
    // across shared-memory banks, matching the independent GSQ tile layout.
    #[inline(always)]
    fn shared_index(row: usize, word: usize) -> usize {
        row * 32 + (word ^ ((row & 7) * 4))
    }

    // Eight coefficients share one BF16 group scale. Decode one aligned word
    // into four half pairs before writing the contiguous shared-memory group.
    #[inline(always)]
    fn values<const HOST: bool, const HALF: bool, const BITS: u32>(
        packed: &[u8],
        scales: &[u16],
        host: *const u8,
        host_scales: *const u16,
        index: usize,
    ) -> [u32; 4] {
        let at = index / (8 / BITS as usize);
        let ptr = if !HOST || at < packed.len() {
            unsafe { packed.as_ptr().add(at) }
        } else {
            unsafe { host.add(at - packed.len()) }
        };
        let word = if BITS == 4 {
            unsafe { ptr.cast::<u32>().read() }
        } else {
            (unsafe { ptr.cast::<u16>().read() }) as u32
        };
        let scale_at = index / 128;
        let scale = if !HOST || scale_at < scales.len() {
            unsafe { *scales.get_unchecked(scale_at) }
        } else {
            unsafe { *host_scales.add(scale_at - scales.len()) }
        };
        if HALF && BITS == 2 {
            let pair = super::super::moe_half::Scale::new(scale as u32);
            return [
                pair.select(word as u32),
                pair.select((word >> 4) as u32),
                pair.select((word >> 8) as u32),
                pair.select((word >> 12) as u32),
            ];
        }
        let scale = f32::from_bits((scale as u32) << 16);
        let mut result = [0; 4];
        let mut j = 0;
        while j < 4 {
            let a = (((word >> (j * 2 * BITS)) & ((1 << BITS) - 1)) as i32 - (1 << (BITS - 1)))
                as f32
                * scale;
            let b = (((word >> ((j * 2 + 1) * BITS)) & ((1 << BITS) - 1)) as i32
                - (1 << (BITS - 1))) as f32
                * scale;
            result[j as usize] = cuda_device::convert::cvt_f16x2_f32(a, b);
            j += 1;
        }
        result
    }

    #[inline(always)]
    fn register_half<const HALF: bool>(
        weights: &[u8],
        scales: &[u16],
        rows: usize,
        columns: usize,
        expert: usize,
        tile: usize,
        k: usize,
        lane: usize,
    ) -> [u32; 4] {
        if HALF {
            return super::super::moe_half::fragment(
                weights, scales, rows, columns, expert, tile, k, lane,
            );
        }
        let bf16 = super::super::moe_register_math::fragment(
            weights, scales, rows, columns, expert, tile, k, lane,
        );
        let mut result = [0; 4];
        let mut i = 0;
        while i < 4 {
            let (a, b) = cuda_device::convert::cvt_f32x2_bf16x2(bf16[i]);
            result[i] = cuda_device::convert::cvt_f16x2_f32(a, b);
            i += 1;
        }
        result
    }

    #[kernel]
    pub unsafe fn matmul<
        const GATE: bool,
        const HOST: bool,
        const REGISTER: bool,
        const PACKED: bool,
        const HALF: bool,
        const STATIC_STORE: bool,
        const BITS: u32,
        const GROUPS: usize,
    >(
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
        packed_input: &[u32],
        output: *mut f32,
    ) {
        let tile = if REGISTER {
            static mut INPUT_TILE: cuda_device::SharedArray<u32, 1024, 16> =
                cuda_device::SharedArray::UNINIT;
            unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut INPUT_TILE) }
        } else if GATE && GROUPS == 16 {
            static mut WIDE_GATE: cuda_device::SharedArray<u32, 8192, 16> =
                cuda_device::SharedArray::UNINIT;
            unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut WIDE_GATE) }
        } else if GATE && GROUPS == 8 {
            static mut MEDIUM_GATE: cuda_device::SharedArray<u32, 6144, 16> =
                cuda_device::SharedArray::UNINIT;
            unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut MEDIUM_GATE) }
        } else if GATE {
            static mut GATE_TILE: cuda_device::SharedArray<u32, 5120, 16> =
                cuda_device::SharedArray::UNINIT;
            unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut GATE_TILE) }
        } else if GROUPS == 16 {
            static mut WIDE_DOWN: cuda_device::SharedArray<u32, 6144, 16> =
                cuda_device::SharedArray::UNINIT;
            unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut WIDE_DOWN) }
        } else if GROUPS == 8 {
            static mut MEDIUM_DOWN: cuda_device::SharedArray<u32, 4096, 16> =
                cuda_device::SharedArray::UNINIT;
            unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut MEDIUM_DOWN) }
        } else {
            static mut DOWN_TILE: cuda_device::SharedArray<u32, 3072, 16> =
                cuda_device::SharedArray::UNINIT;
            unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut DOWN_TILE) }
        };
        let input_base = if REGISTER {
            0
        } else if GATE {
            4096
        } else {
            2048
        };
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
        // GROUPS*8 disjoint choices; row/choice tails are zero padded.
        let mut sums = [[0.0f32; 4]; GROUPS];
        let mut uppers = [[0.0f32; 4]; GROUPS];
        let mut col = 0;
        while col < cols {
            if !REGISTER {
                let mut vector = local;
                while vector < 512 {
                    let row = row_base + vector / 8;
                    let column = col + vector % 8 * 8;
                    let index = (expert * rows + row) * cols + column;
                    let gate = if row < rows {
                        values::<HOST, HALF, BITS>(
                            weights,
                            scales,
                            host_weights,
                            host_scales,
                            index,
                        )
                    } else {
                        [0; 4]
                    };
                    let upper = if GATE && row < rows {
                        values::<HOST, HALF, BITS>(up, up_scales, host_up, host_up_scales, index)
                    } else {
                        [0; 4]
                    };
                    let at = shared_index(vector / 8, vector % 8 * 4);
                    let mut j = 0;
                    #[unroll]
                    while j < 4 {
                        unsafe {
                            *tile.add(at + j) = gate[j];
                            if GATE {
                                *tile.add(2048 + at + j) = upper[j];
                            }
                        }
                        j += 1;
                    }
                    vector += 128;
                }
            }
            let mut vector = local;
            while vector < GROUPS * 128 {
                let slot = token_base + vector / 16;
                let column = col + vector % 16 * 4;
                let words = if slot < counts[expert] as usize {
                    let choice = choices[expert * batch + slot] as usize;
                    let source = if GATE { choice / 8 } else { choice };
                    if PACKED {
                        // Complete four-value groups preserve eight-byte alignment.
                        let pair = unsafe {
                            packed_input
                                .as_ptr()
                                .add((source * cols + column) / 2)
                                .cast::<u64>()
                                .read()
                        };
                        [pair as u32, (pair >> 32) as u32]
                    } else {
                        let x = unsafe {
                            input
                                .as_ptr()
                                .add(source * cols + column)
                                .cast::<Float4>()
                                .read()
                                .values
                        };
                        [
                            cuda_device::convert::cvt_f16x2_f32(x[0], x[1]),
                            cuda_device::convert::cvt_f16x2_f32(x[2], x[3]),
                        ]
                    }
                } else {
                    [0; 2]
                };
                let at = input_base + shared_index(vector / 16, vector % 16 * 2);
                unsafe {
                    *tile.add(at) = words[0];
                    *tile.add(at + 1) = words[1];
                }
                vector += 128;
            }
            thread::sync_threads();
            let mut k = 0;
            while k < 64 {
                unsafe {
                    let offset = shared_index(warp * 16 + lane % 16, lane / 16 * 4 + k / 2);
                    let a = if REGISTER {
                        register_half::<HALF>(
                            weights,
                            scales,
                            rows,
                            cols,
                            expert,
                            row_base / 16 + warp,
                            col + k,
                            lane,
                        )
                    } else {
                        cuda_device::wmma::ldmatrix_x4(tile.add(offset))
                    };
                    let upper = if GATE {
                        if REGISTER {
                            register_half::<HALF>(
                                up,
                                up_scales,
                                rows,
                                cols,
                                expert,
                                row_base / 16 + warp,
                                col + k,
                                lane,
                            )
                        } else {
                            cuda_device::wmma::ldmatrix_x4(tile.add(2048 + offset))
                        }
                    } else {
                        [0; 4]
                    };
                    let mut j = 0;
                    #[unroll]
                    while j < GROUPS {
                        let b = cuda_device::wmma::ldmatrix_x2(tile.add(
                            input_base + shared_index(j * 8 + lane % 8, lane / 8 % 2 * 4 + k / 2),
                        ));
                        if BITS == 4 {
                            // B's two packed pairs represent K offsets0 and8
                            // for the token owned by this lane's four-lane group.
                            let slot = token_base + j * 8 + lane / 4;
                            let low = if slot < counts[expert] as usize {
                                let choice = choices[expert * batch + slot] as usize;
                                let source = if GATE { choice / 8 } else { choice };
                                let offset = source * cols + col + k + (lane % 4) * 2;
                                if PACKED {
                                    let base = input.len() / 2 + offset / 2;
                                    [packed_input[base], packed_input[base + 4]]
                                } else {
                                    super::super::moe_residual::fragment(input, offset)
                                }
                            } else {
                                [0; 2]
                            };
                            sums[j] = cuda_device::wmma::mma_m16n8k16_f32_f16(sums[j], a, low);
                            if GATE {
                                uppers[j] =
                                    cuda_device::wmma::mma_m16n8k16_f32_f16(uppers[j], upper, low);
                            }
                        }
                        sums[j] = cuda_device::wmma::mma_m16n8k16_f32_f16(sums[j], a, b);
                        if GATE {
                            uppers[j] =
                                cuda_device::wmma::mma_m16n8k16_f32_f16(uppers[j], upper, b);
                        }
                        j += 1;
                    }
                }
                k += 16;
            }
            thread::sync_threads();
            col += 64;
        }
        let row = row_base + warp * 16 + lane / 4;
        let token = token_base + lane % 4 * 2;
        let mut j = 0;
        #[unroll]
        while j < GROUPS {
            unsafe {
                store::<GATE, STATIC_STORE>(
                    rows,
                    batch,
                    expert,
                    row,
                    token + j * 8,
                    sums[j],
                    uppers[j],
                    counts,
                    choices,
                    output,
                );
            }
            j += 1;
        }
    }

    #[inline(always)]
    unsafe fn store<const GATE: bool, const STATIC_STORE: bool>(
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
        if STATIC_STORE {
            // Literal indices keep the final MMA fragments in SSA form. A
            // dynamic loop here makes the compiler materialize both arrays
            // in thread-local memory despite no register-allocator spills.
            macro_rules! component {
                ($i:literal) => {{
                    let r = row + if $i >= 2 { 8 } else { 0 };
                    let slot = token + $i % 2;
                    if r < rows && slot < counts[expert] as usize {
                        let choice = choices[expert * batch + slot] as usize;
                        let value = if GATE {
                            sums[$i] / (1.0 + (-sums[$i]).exp()) * upper[$i]
                        } else {
                            sums[$i]
                        };
                        unsafe {
                            *output.add(choice * rows + r) = value;
                        }
                    }
                }};
            }
            component!(0);
            component!(1);
            component!(2);
            component!(3);
            return;
        }
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
