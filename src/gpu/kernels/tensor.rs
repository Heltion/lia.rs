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

    /// Decode packed GGUF weights once for the entire prefill batch.
    #[kernel]
    pub unsafe fn unpack_matrix<const KIND: u32, const BYTES: usize>(
        weights: &[u8],
        output: *mut u32,
        elements: usize,
    ) {
        let i = thread::index_1d().get() * 8;
        if i < elements {
            let dest = i / 2;
            if KIND == 0 || KIND == 30 {
                let mut j = 0;
                while j < 4 {
                    let a = crate::quant::decode_value(
                        KIND,
                        &weights[(i + 2 * j) * BYTES..(i + 2 * j + 1) * BYTES],
                        0,
                    );
                    let b = crate::quant::decode_value(
                        KIND,
                        &weights[(i + 2 * j + 1) * BYTES..(i + 2 * j + 2) * BYTES],
                        0,
                    );
                    unsafe {
                        *output.add(dest + j) = cuda_device::convert::cvt_f16x2_f32(a, b);
                    }
                    j += 1;
                }
            } else {
                let block = i / 256;
                let data = unsafe {
                    core::slice::from_raw_parts(weights.as_ptr().add(block * BYTES), BYTES)
                };
                let (lo, hi, scale, offset) =
                    unsafe { crate::quant::packed_eight_aligned(KIND, data, i % 256) };
                unsafe {
                    *output.add(dest) = pair(lo, 0, scale, offset);
                    *output.add(dest + 1) = pair(lo, 16, scale, offset);
                    *output.add(dest + 2) = pair(hi, 0, scale, offset);
                    *output.add(dest + 3) = pair(hi, 16, scale, offset);
                }
            }
        }
    }

    #[kernel]
    pub unsafe fn pack_input(input: &[f32], output: *mut u32) {
        let i = thread::index_1d().get();
        if i < input.len() / 2 {
            // SAFETY: the host allocates one packed word per complete pair.
            unsafe {
                *output.add(i) =
                    cuda_device::convert::cvt_f16x2_f32(input[2 * i], input[2 * i + 1]);
            }
        }
    }

    /// Reuse 128-row operand tiles across eight warps. Each warp owns four
    /// 16-row fragments and four 8-token fragments with independent f32 sums.
    #[kernel]
    pub unsafe fn prefill_matrix(
        cols: usize,
        rows: usize,
        batch: usize,
        weights: &[u32],
        input: &[u32],
        output: *mut f32,
    ) {
        static mut TILE: cuda_device::SharedArray<u32, 9216, 16> = cuda_device::SharedArray::UNINIT;
        let tid = thread::index_1d().get() as u32;
        let cols = cols as u32;
        let rows = rows as u32;
        let batch = batch as u32;
        let local = tid % 256;
        let lane = local % 32;
        let warp = local / 32;
        let tile_id = tid / 256;
        let row_base = (tile_id % rows.div_ceil(128)) * 128;
        let token_base = (tile_id / rows.div_ceil(128)) * 128;
        // SAFETY: complete CTAs cooperatively fill disjoint shared words and
        // reach both barriers. Invalid tail rows and tokens are zero padded.
        let tile = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut TILE) };
        let mut sum00 = [0.0f32; 4];
        let mut sum01 = [0.0f32; 4];
        let mut sum02 = [0.0f32; 4];
        let mut sum03 = [0.0f32; 4];
        let mut sum10 = [0.0f32; 4];
        let mut sum11 = [0.0f32; 4];
        let mut sum12 = [0.0f32; 4];
        let mut sum13 = [0.0f32; 4];
        let mut sum20 = [0.0f32; 4];
        let mut sum21 = [0.0f32; 4];
        let mut sum22 = [0.0f32; 4];
        let mut sum23 = [0.0f32; 4];
        let mut sum30 = [0.0f32; 4];
        let mut sum31 = [0.0f32; 4];
        let mut sum32 = [0.0f32; 4];
        let mut sum33 = [0.0f32; 4];
        let mut col = 0;
        while col < cols {
            let mut i = local * 4;
            while i < 4096 {
                let row = row_base + i / 32;
                // SAFETY: complete aligned groups inside validated operand rows.
                let w = if row < rows {
                    unsafe {
                        weights
                            .as_ptr()
                            .add((row * (cols / 2) + col / 2 + i % 32) as usize)
                            .cast::<Words4>()
                            .read()
                    }
                } else {
                    Words4 { data: [0; 4] }
                };
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
                // SAFETY: padded shared rows preserve 16-byte alignment, and
                // each thread writes one distinct complete four-word group.
                unsafe {
                    tile.add(((i / 32) * 36 + i % 32) as usize)
                        .cast::<Words4>()
                        .write(w);
                    tile.add((4608 + (i / 32) * 36 + i % 32) as usize)
                        .cast::<Words4>()
                        .write(x);
                }
                i += 1024;
            }
            thread::sync_threads();
            let group = (warp / 4) * 64;
            let token_group = (warp % 4) * 32;
            let mut j = 0;
            while j < 64 {
                // SAFETY: each fragment address is inside the completed padded
                // tile and matches MMA's lane-to-register layout.
                unsafe {
                    let b0 = cuda_device::wmma::ldmatrix_x2(tile.add(
                        (4608 + (token_group + lane % 8) * 36 + ((lane / 8) % 2) * 4 + j / 2)
                            as usize,
                    ));
                    let b1 = cuda_device::wmma::ldmatrix_x2(tile.add(
                        (4608 + (token_group + 8 + lane % 8) * 36 + ((lane / 8) % 2) * 4 + j / 2)
                            as usize,
                    ));
                    let b2 = cuda_device::wmma::ldmatrix_x2(tile.add(
                        (4608 + (token_group + 16 + lane % 8) * 36 + ((lane / 8) % 2) * 4 + j / 2)
                            as usize,
                    ));
                    let b3 = cuda_device::wmma::ldmatrix_x2(tile.add(
                        (4608 + (token_group + 24 + lane % 8) * 36 + ((lane / 8) % 2) * 4 + j / 2)
                            as usize,
                    ));
                    let a0 = cuda_device::wmma::ldmatrix_x4(
                        tile.add(((group + lane % 16) * 36 + (lane / 16) * 4 + j / 2) as usize),
                    );
                    sum00 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum00, a0, b0);
                    sum01 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum01, a0, b1);
                    sum02 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum02, a0, b2);
                    sum03 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum03, a0, b3);
                    let a1 =
                        cuda_device::wmma::ldmatrix_x4(tile.add(
                            ((group + 16 + lane % 16) * 36 + (lane / 16) * 4 + j / 2) as usize,
                        ));
                    sum10 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum10, a1, b0);
                    sum11 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum11, a1, b1);
                    sum12 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum12, a1, b2);
                    sum13 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum13, a1, b3);
                    let a2 =
                        cuda_device::wmma::ldmatrix_x4(tile.add(
                            ((group + 32 + lane % 16) * 36 + (lane / 16) * 4 + j / 2) as usize,
                        ));
                    sum20 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum20, a2, b0);
                    sum21 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum21, a2, b1);
                    sum22 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum22, a2, b2);
                    sum23 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum23, a2, b3);
                    let a3 =
                        cuda_device::wmma::ldmatrix_x4(tile.add(
                            ((group + 48 + lane % 16) * 36 + (lane / 16) * 4 + j / 2) as usize,
                        ));
                    sum30 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum30, a3, b0);
                    sum31 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum31, a3, b1);
                    sum32 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum32, a3, b2);
                    sum33 = cuda_device::wmma::mma_m16n8k16_f32_f16(sum33, a3, b3);
                }
                j += 16;
            }
            thread::sync_threads();
            col += 64;
        }
        let row = row_base + (warp / 4) * 64 + lane / 4;
        let token = token_base + (warp % 4) * 32 + (lane % 4) * 2;
        // SAFETY: each warp/lane owns disjoint outputs; every padded extent is
        // guarded before writing the token-major result.
        unsafe {
            store(rows, batch, row, token, sum00, output);
            store(rows, batch, row, token + 8, sum01, output);
            store(rows, batch, row, token + 16, sum02, output);
            store(rows, batch, row, token + 24, sum03, output);
            store(rows, batch, row + 16, token, sum10, output);
            store(rows, batch, row + 16, token + 8, sum11, output);
            store(rows, batch, row + 16, token + 16, sum12, output);
            store(rows, batch, row + 16, token + 24, sum13, output);
            store(rows, batch, row + 32, token, sum20, output);
            store(rows, batch, row + 32, token + 8, sum21, output);
            store(rows, batch, row + 32, token + 16, sum22, output);
            store(rows, batch, row + 32, token + 24, sum23, output);
            store(rows, batch, row + 48, token, sum30, output);
            store(rows, batch, row + 48, token + 8, sum31, output);
            store(rows, batch, row + 48, token + 16, sum32, output);
            store(rows, batch, row + 48, token + 24, sum33, output);
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
