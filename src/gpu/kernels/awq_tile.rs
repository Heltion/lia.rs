//! Shared K64 AWQ arithmetic; token tile width controls packed-weight reuse.

use cuda_device::{convert, kernel, thread, wmma};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub unsafe fn awq_fp16_wide(
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
        static mut TILE: cuda_device::SharedArray<u32, 3456, 16> = cuda_device::SharedArray::UNINIT;
        unsafe {
            let tile = cuda_device::SharedArray::as_raw_mut_ptr(&raw mut TILE);
            compute::<64, 32, 128, false, false>(
                cols,
                rows,
                group_size,
                batch,
                weights,
                zeros,
                scales,
                input,
                output,
                tile,
                thread::index_1d().get() as u32,
                1,
            );
        }
    }
    #[kernel]
    pub unsafe fn awq_fp16_square(
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
        static mut TILE: cuda_device::SharedArray<u32, 4608, 16> = cuda_device::SharedArray::UNINIT;
        unsafe {
            let tile = cuda_device::SharedArray::as_raw_mut_ptr(&raw mut TILE);
            compute::<64, 64, 256, false, false>(
                cols,
                rows,
                group_size,
                batch,
                weights,
                zeros,
                scales,
                input,
                output,
                tile,
                thread::index_1d().get() as u32,
                1,
            );
        }
    }
    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    struct Gains {
        values: [f32; 8],
    }
    #[repr(C, align(8))]
    #[derive(Clone, Copy)]
    struct Pair {
        values: [f32; 2],
    }

    #[inline(always)]
    pub(crate) unsafe fn compute<
        const TOKENS: u32,
        const ROW_TILE: u32,
        const THREADS: u32,
        const SPLIT: bool,
        const GATEUP: bool,
    >(
        cols: usize,
        rows: usize,
        group_size: usize,
        batch: usize,
        weights: &[u32],
        zeros: &[u32],
        scales: &[f32],
        input: &[f32],
        output: *mut f32,
        tile: *mut u32,
        tid: u32,
        parts: u32,
    ) {
        // The checked host launch bounds every buffer and the flattened grid
        // to u32. Widen only the final element offsets for pointer arithmetic.
        let (cols, rows, group_size, batch) =
            (cols as u32, rows as u32, group_size as u32, batch as u32);
        let part = if SPLIT { tid / THREADS % parts } else { 0 };
        let tid = if SPLIT {
            tid / THREADS / parts * THREADS + tid % THREADS
        } else {
            tid
        };
        let local = tid % THREADS;
        let lane = local % 32;
        let warp = local / 32;
        let n_warps = THREADS / 32 / (ROW_TILE / 16);
        let token_tiles = batch.div_ceil(TOKENS);
        let row_base = tid / THREADS / token_tiles * ROW_TILE;
        let token_base = tid / THREADS % token_tiles * TOKENS;
        let word_row = row_base / 8 + warp;
        let ah = tile;
        let bh = unsafe { tile.add((ROW_TILE * 36) as usize) };
        let groups = cols / group_size;
        let width = cols / parts;
        let mut col = part * width;
        let end = col + width;
        let mut group = col / group_size;
        let mut next_group = (group + 1) * group_size;
        let mut sum0 = [0.0f32; 4];
        let mut sum1 = [0.0f32; 4];
        let mut sum2 = [0.0f32; 4];
        let mut sum3 = [0.0f32; 4];
        while col < end {
            if col == next_group {
                group += 1;
                next_group += group_size;
            }
            let k = col + lane * 2;
            let (packed, zero, gains) =
                if word_row < rows / 8 && (THREADS / 32 <= ROW_TILE / 8 || warp < ROW_TILE / 8) {
                    unsafe {
                        (
                            weights
                                .as_ptr()
                                .add((word_row * cols + k) as usize)
                                .cast::<u64>()
                                .read(),
                            zeros[(word_row * groups + group) as usize],
                            scales
                                .as_ptr()
                                .add((group * rows + word_row * 8) as usize)
                                .cast::<Gains>()
                                .read()
                                .values,
                        )
                    }
                } else {
                    (0, 0, [0.0; 8])
                };
            // Literal rows expose independent loads without relying on loop-unroll lowering.
            macro_rules! weight_row {
                ($row:literal) => {{
                    let r = $row;
                    let shift = ((r % 2) * 4 + r / 2) * 4;
                    let z = ((zero >> shift) & 15) as i32;
                    let a = (((packed as u32) >> shift) & 15) as i32;
                    let b = ((((packed >> 32) as u32) >> shift) & 15) as i32;
                    let pair = convert::cvt_f16x2_f32(
                        (a - z) as f32 * gains[r as usize],
                        (b - z) as f32 * gains[r as usize],
                    );
                    unsafe {
                        ah.add(((warp * 8 + r) * 36 + lane) as usize).write(pair);
                    }
                }};
            }
            if THREADS / 32 <= ROW_TILE / 8 || warp < ROW_TILE / 8 {
                weight_row!(0u32);
                weight_row!(1u32);
                weight_row!(2u32);
                weight_row!(3u32);
                weight_row!(4u32);
                weight_row!(5u32);
                weight_row!(6u32);
                weight_row!(7u32);
            }
            macro_rules! input_row {
                ($row:literal) => {{
                    let r = $row;
                    let token = token_base + warp * (TOKENS / (THREADS / 32)) + r;
                    let x = if token < batch {
                        unsafe {
                            input
                                .as_ptr()
                                .add((token * cols + k) as usize)
                                .cast::<Pair>()
                                .read()
                                .values
                        }
                    } else {
                        [0., 0.]
                    };
                    unsafe {
                        bh.add(((warp * (TOKENS / (THREADS / 32)) + r) * 36 + lane) as usize)
                            .write(convert::cvt_f16x2_f32(x[0], x[1]));
                    }
                }};
            }
            input_row!(0u32);
            input_row!(1u32);
            input_row!(2u32);
            input_row!(3u32);
            input_row!(4u32);
            input_row!(5u32);
            input_row!(6u32);
            input_row!(7u32);
            if TOKENS == 64 && THREADS == 128 {
                input_row!(8u32);
                input_row!(9u32);
                input_row!(10u32);
                input_row!(11u32);
                input_row!(12u32);
                input_row!(13u32);
                input_row!(14u32);
                input_row!(15u32);
            }
            thread::sync_threads();
            let mut k = 0;
            while k < 64 {
                let a = ((warp / n_warps) * 16 + lane % 16) * 36 + (lane / 16) * 4 + k / 2;
                let b = ((warp % n_warps) * (TOKENS / n_warps) + lane % 8) * 36
                    + ((lane / 8) % 2) * 4
                    + k / 2;
                unsafe {
                    let high_a = wmma::ldmatrix_x4(ah.add(a as usize));
                    sum0 = wmma::mma_m16n8k16_f32_f16(
                        sum0,
                        high_a,
                        wmma::ldmatrix_x2(bh.add(b as usize)),
                    );
                    sum1 = wmma::mma_m16n8k16_f32_f16(
                        sum1,
                        high_a,
                        wmma::ldmatrix_x2(bh.add((b + 288) as usize)),
                    );
                    if TOKENS == 64 {
                        sum2 = wmma::mma_m16n8k16_f32_f16(
                            sum2,
                            high_a,
                            wmma::ldmatrix_x2(bh.add((b + 576) as usize)),
                        );
                        sum3 = wmma::mma_m16n8k16_f32_f16(
                            sum3,
                            high_a,
                            wmma::ldmatrix_x2(bh.add((b + 864) as usize)),
                        );
                    }
                }
                k += 16;
            }
            thread::sync_threads();
            col += 64;
        }
        let row = row_base + (warp / n_warps) * 16 + lane / 4;
        let token = token_base + (warp % n_warps) * (TOKENS / n_warps) + lane % 4 * 2;
        if GATEUP {
            let activation_row = row / 16 * 8 + row % 16;
            unsafe {
                super::super::awq_gateup::device::finish(
                    rows / 2,
                    batch,
                    activation_row,
                    token,
                    sum0,
                    output,
                );
                super::super::awq_gateup::device::finish(
                    rows / 2,
                    batch,
                    activation_row,
                    token + 8,
                    sum1,
                    output,
                );
                if TOKENS == 64 {
                    super::super::awq_gateup::device::finish(
                        rows / 2,
                        batch,
                        activation_row,
                        token + 16,
                        sum2,
                        output,
                    );
                    super::super::awq_gateup::device::finish(
                        rows / 2,
                        batch,
                        activation_row,
                        token + 24,
                        sum3,
                        output,
                    );
                }
            }
            return;
        }
        unsafe {
            store(
                rows as usize,
                batch as usize,
                row as usize,
                token as usize,
                sum0,
                output,
                parts as usize,
                part as usize,
            );
            store(
                rows as usize,
                batch as usize,
                row as usize,
                (token + 8) as usize,
                sum1,
                output,
                parts as usize,
                part as usize,
            );
            if TOKENS == 64 {
                store(
                    rows as usize,
                    batch as usize,
                    row as usize,
                    (token + 16) as usize,
                    sum2,
                    output,
                    parts as usize,
                    part as usize,
                );
                store(
                    rows as usize,
                    batch as usize,
                    row as usize,
                    (token + 24) as usize,
                    sum3,
                    output,
                    parts as usize,
                    part as usize,
                );
            }
        }
    }
    #[inline(always)]
    unsafe fn store(
        rows: usize,
        batch: usize,
        row: usize,
        token: usize,
        sum: [f32; 4],
        output: *mut f32,
        parts: usize,
        part: usize,
    ) {
        if row < rows && token < batch {
            unsafe {
                *output.add((token * parts + part) * rows + row) = sum[0];
            }
        }
        if row < rows && token + 1 < batch {
            unsafe {
                *output.add(((token + 1) * parts + part) * rows + row) = sum[1];
            }
        }
        if row + 8 < rows && token < batch {
            unsafe {
                *output.add((token * parts + part) * rows + row + 8) = sum[2];
            }
        }
        if row + 8 < rows && token + 1 < batch {
            unsafe {
                *output.add(((token + 1) * parts + part) * rows + row + 8) = sum[3];
            }
        }
    }
}
