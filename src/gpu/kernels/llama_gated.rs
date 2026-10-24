//! Dual BF16 projections share packed input tiles and fuse the SiLU epilogue.
use cuda_device::{kernel, launch_bounds, thread, warp};
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

    // Permute aligned eight-half groups across banks instead of padding rows.
    // Each ldmatrix address still names an unchanged contiguous eight-half group.
    #[inline(always)]
    fn shared_index(row: u32, word: u32) -> usize {
        (row * 32 + (word ^ ((row & 7) * 4))) as usize
    }

    /// A 64-by-64 tile retains both projections until the nonlinear epilogue.
    #[kernel]
    #[launch_bounds(256, 4)]
    pub unsafe fn llama_gated<const HALF: bool>(
        cols: usize,
        rows: usize,
        batch: usize,
        gate_weights: &[u8],
        up_weights: &[u8],
        input: *const u32,
        output: *mut u32,
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
        let token_tiles = batch.div_ceil(64);
        let row_base = tile_id / token_tiles * 64;
        let token_base = tile_id % token_tiles * 64;
        // SAFETY: every CTA fills disjoint words and reaches both barriers;
        // all row/token tails are padded before the warp matrix loads.
        static mut TILE: cuda_device::SharedArray<u32, 6144, 16> = cuda_device::SharedArray::UNINIT;
        let tile = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut TILE) };
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
            let mut vec = local;
            while vec < 1024 {
                let plane = vec / 512;
                let index = vec % 512;
                let row = row_base + index / 8;
                let k = (index % 8) * 8;
                let dest = plane as usize * 2048 + shared_index(index / 8, k / 2);
                let weights = if plane == 0 { gate_weights } else { up_weights };
                let packed = if row < rows {
                    unsafe {
                        weights
                            .as_ptr()
                            .add(((row * cols + col + k) * 2) as usize)
                            .cast::<Words4>()
                            .read()
                    }
                } else {
                    Words4 { data: [0; 4] }
                };
                let convert = |word: u32| {
                    cuda_device::convert::cvt_f16x2_f32(
                        f32::from_bits(word << 16),
                        f32::from_bits(word & 0xffff_0000),
                    )
                };
                let half = if HALF {
                    packed
                } else {
                    Words4 {
                        data: [
                            convert(packed.data[0]),
                            convert(packed.data[1]),
                            convert(packed.data[2]),
                            convert(packed.data[3]),
                        ],
                    }
                };
                unsafe {
                    tile.add(dest).cast::<Words4>().write(half);
                }
                vec += 256;
            }
            let mut i = local * 4;
            while i < 64 * 32 {
                let token = token_base + i / 32;
                let x = if token < batch {
                    unsafe {
                        input
                            .add((token * (cols / 2) + col / 2 + i % 32) as usize)
                            .cast::<Words4>()
                            .read()
                    }
                } else {
                    Words4 { data: [0; 4] }
                };
                unsafe {
                    tile.add(4096 + shared_index(i / 32, i % 32))
                        .cast::<Words4>()
                        .write(x);
                }
                i += 1024;
            }
            thread::sync_threads();
            let group = (warp / 4) * 32;
            let token_group = (warp % 4) * (64 / 4);
            let mut j = 0usize;
            #[unroll]
            while j < 4 {
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

                    let u0 = cuda_device::wmma::ldmatrix_x4(tile.add(
                        2048 + shared_index(group + lane % 16, (lane / 16) * 4 + j as u32 * 8),
                    ));
                    let u1 = cuda_device::wmma::ldmatrix_x4(tile.add(
                        2048 + shared_index(group + 16 + lane % 16, (lane / 16) * 4 + j as u32 * 8),
                    ));
                    macro_rules! product {
                        ($offset:literal, $g0:ident, $g1:ident, $u0:ident, $u1:ident) => {{
                            let b = cuda_device::wmma::ldmatrix_x2(tile.add(
                                4096 + shared_index(
                                    token_group + $offset + lane % 8,
                                    ((lane / 8) % 2) * 4 + j as u32 * 8,
                                ),
                            ));
                            $g0 = cuda_device::wmma::mma_m16n8k16_f32_f16($g0, a0, b);
                            $g1 = cuda_device::wmma::mma_m16n8k16_f32_f16($g1, a1, b);
                            $u0 = cuda_device::wmma::mma_m16n8k16_f32_f16($u0, u0, b);
                            $u1 = cuda_device::wmma::mma_m16n8k16_f32_f16($u1, u1, b);
                        }};
                    }
                    product!(0, sum00, sum10, sum02, sum12);
                    product!(8, sum01, sum11, sum03, sum13);
                }
                j += 1;
            }
            thread::sync_threads();
            col += 64;
        }
        let row = row_base + (warp / 4) * 32 + lane / 4;
        let token = token_base + (warp % 4) * (64 / 4) + (lane % 4) * 2;
        // SAFETY: disjoint lane/warp outputs with explicit tail guards.
        unsafe {
            store(rows, batch, row, token, lane, sum00, sum02, output);
            store(rows, batch, row + 16, token, lane, sum10, sum12, output);
            store(rows, batch, row, token + 8, lane, sum01, sum03, output);
            store(rows, batch, row + 16, token + 8, lane, sum11, sum13, output);
        }
    }

    #[inline(always)]
    unsafe fn store(
        rows: u32,
        batch: u32,
        row: u32,
        token: u32,
        lane: u32,
        gate: [f32; 4],
        up: [f32; 4],
        output: *mut u32,
    ) {
        let a = up[0] * gate[0] / (1.0 + (-gate[0]).exp());
        let b = up[1] * gate[1] / (1.0 + (-gate[1]).exp());
        let c = up[2] * gate[2] / (1.0 + (-gate[2]).exp());
        let d = up[3] * gate[3] / (1.0 + (-gate[3]).exp());
        // Adjacent matrix rows reside in lanes four apart. All lanes take part
        // before the even row owns the complete aligned half pair.
        let an = warp::shuffle_f32_sync(0xffff_ffff, a, lane ^ 4);
        let bn = warp::shuffle_f32_sync(0xffff_ffff, b, lane ^ 4);
        let cn = warp::shuffle_f32_sync(0xffff_ffff, c, lane ^ 4);
        let dn = warp::shuffle_f32_sync(0xffff_ffff, d, lane ^ 4);
        if row.is_multiple_of(2) {
            unsafe {
                if row + 1 < rows && token < batch {
                    *output.add((token * rows + row) as usize / 2) =
                        cuda_device::convert::cvt_f16x2_f32(a, an);
                }
                if row + 1 < rows && token + 1 < batch {
                    *output.add(((token + 1) * rows + row) as usize / 2) =
                        cuda_device::convert::cvt_f16x2_f32(b, bn);
                }
                if row + 9 < rows && token < batch {
                    *output.add((token * rows + row + 8) as usize / 2) =
                        cuda_device::convert::cvt_f16x2_f32(c, cn);
                }
                if row + 9 < rows && token + 1 < batch {
                    *output.add(((token + 1) * rows + row + 8) as usize / 2) =
                        cuda_device::convert::cvt_f16x2_f32(d, dn);
                }
            }
        }
    }
}
