//! Prepare independent head/chunk coefficients once for all value shards.
use super::delta_mma::{product, put, scalar};
use super::delta_tiles::{BETA, LOWER, RATIO, invert, matrices, prepare};
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub unsafe fn prepare_chunks(
        k_heads: usize,
        v_heads: usize,
        qk: &[f32],
        decay: &[f32],
        beta: &[f32],
        output: *mut u32,
    ) {
        static mut TILE: cuda_device::SharedArray<u32, 11440, 16> =
            cuda_device::SharedArray::UNINIT;
        let tid = thread::index_1d().get() as u32;
        let local = tid % 256;
        let warp = local / 32;
        let lane = local % 32;
        let batch = decay.len() as u32 / v_heads as u32;
        let chunks = batch.div_ceil(16);
        let block = tid / 256;
        let head = block / chunks;
        let start = (block % chunks) * 16;
        let valid = (batch - start).min(16);
        // SAFETY: one CTA owns an entire head/chunk and its disjoint output region.
        let tile = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut TILE) };
        let [q, k, p, inverse, _, _, _] = unsafe { matrices(tile) };
        // The stored coefficient planes include alignment padding. Initialize that
        // padding explicitly instead of exporting uninitialized shared words.
        let mut i = local;
        while i < 5120 {
            unsafe { tile.add(i as usize).write(0) };
            i += 256;
        }
        thread::sync_threads();
        unsafe {
            prepare(
                tile,
                q,
                k,
                head,
                k_heads as u32,
                v_heads as u32,
                start,
                valid,
                local,
                qk,
                decay,
                beta,
            );
        }
        if warp < 4 {
            let col = (warp % 2) * 8;
            let gram = warp < 2;
            let values = unsafe {
                product::<128, false, false>([0.0; 4], if gram { k } else { q }, k, 0, col, lane)
            };
            let mut coefficients = [0.0; 4];
            let mut j = 0;
            #[unroll]
            while j < 4 {
                let row = lane / 4 + (j / 2) * 8;
                let c = col + (lane % 4) * 2 + j % 2;
                let value = if row < valid && c < valid && c <= row {
                    values[j as usize] * unsafe { scalar(tile, RATIO + row * 16 + c) }
                } else {
                    0.0
                };
                if gram {
                    unsafe {
                        put(
                            tile,
                            LOWER + row * 16 + c,
                            if c < row {
                                value * scalar(tile, BETA + row)
                            } else {
                                0.0
                            },
                        );
                    }
                } else {
                    coefficients[j as usize] = value;
                }
                j += 1;
            }
            if !gram {
                unsafe {
                    p.fragment(0, col, lane, coefficients);
                }
            }
        }
        thread::sync_threads();
        unsafe {
            invert(tile, inverse, warp, lane);
        }
        thread::sync_threads();
        i = local;
        while i < 1072 {
            let source = if i < 768 { 4352 + i } else { RATIO + i - 768 };
            unsafe {
                output
                    .add((block * 1072 + i) as usize)
                    .write(*tile.add(source as usize));
            }
            i += 256;
        }
    }
}
