//! Test-only PV register caching permits probability storage to be released early.
use cuda_device::{kernel, launch_bounds, launch_contract, thread, wmma};
use cuda_host::cuda_module;

#[inline(always)]
fn probability(row: usize, word: usize) -> usize {
    row * 32 + (word ^ ((row & 7) * 4))
}
#[inline(always)]
fn value(row: usize, word: usize) -> usize {
    row * 128 + (word ^ ((row & 7) * 4))
}

#[cuda_module]
pub(crate) mod device {
    use super::super::flash_mma_math::{Float2, Words4, split};
    use super::*;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain=1,dynamic_shared_range=(0,101376),dynamic_shared_alignment=16)]
    pub unsafe fn pv<const COMPACT: bool>(p: &[f32], v: &[u16], output: *mut f32) {
        let local = thread::index_1d().get();
        let lane = local % 32;
        let w = local / 32;
        let memory = cuda_device::DynamicSharedArray::<u32, 16>::get();
        let probs = unsafe { memory.add(if COMPACT { 24320 } else { 0 }) };
        let values = unsafe { memory.add(if COMPACT { 20224 } else { 1280 }) };
        let mut i = local;
        while i < 512 {
            let row = i / 16;
            let word = i % 16;
            let at = row * 32 + word * 2;
            let (high, low) = if COMPACT {
                (probability(row, word), probability(row, word + 16))
            } else {
                (row * 20 + word, 640 + row * 20 + word)
            };
            unsafe { split(probs.add(high), probs.add(low), p[at], p[at + 1]) };
            i += 256;
        }
        i = local * 4;
        while i < 4096 {
            let word = unsafe { v.as_ptr().cast::<u32>().add(i).cast::<Words4>().read() };
            let at = if COMPACT {
                value(i / 128, i % 128)
            } else {
                i / 128 * 132 + i % 128
            };
            unsafe { values.add(at).cast::<Words4>().write(word) };
            i += 1024;
        }
        thread::sync_threads();
        let mut high = [[0u32; 4]; 2];
        let mut low = [[0u32; 4]; 2];
        let row = w / 4 * 16 + lane % 16;
        let mut step = 0;
        #[unroll]
        while step < 2 {
            let word = step * 8 + lane / 16 * 4;
            let (a, b) = if COMPACT {
                (probability(row, word), probability(row, word + 16))
            } else {
                (row * 20 + word, 640 + row * 20 + word)
            };
            high[step] = unsafe { wmma::ldmatrix_x4(probs.add(a)) };
            low[step] = unsafe { wmma::ldmatrix_x4(probs.add(b)) };
            step += 1;
        }
        thread::sync_threads();
        // A next producer may now overwrite every probability word. Deliberate
        // half NaNs make an accidental post-release reload observable.
        i = local;
        while i < if COMPACT { 1024 } else { 1280 } {
            unsafe { *probs.add(i) = 0x7e00_7e00 };
            i += 256;
        }
        thread::sync_threads();
        let mut output_tile = [[0.0f32; 4]; 8];
        step = 0;
        #[unroll]
        while step < 2 {
            let mut j = 0;
            #[unroll]
            while j < 8 {
                let col = w % 4 * 64 + j * 8;
                let at = if COMPACT {
                    value(step * 16 + lane % 16, col / 2)
                } else {
                    (step * 16 + lane % 16) * 132 + col / 2
                };
                let b = unsafe { wmma::ldmatrix_x2_trans(values.add(at)) };
                output_tile[j] =
                    unsafe { wmma::mma_m16n8k16_f32_f16(output_tile[j], low[step], b) };
                output_tile[j] =
                    unsafe { wmma::mma_m16n8k16_f32_f16(output_tile[j], high[step], b) };
                j += 1;
            }
            step += 1;
        }
        let mut j = 0;
        #[unroll]
        while j < 8 {
            let col = w % 4 * 64 + j * 8 + lane % 4 * 2;
            macro_rules! store {
                ($r:literal) => {{
                    let row = w / 4 * 16 + lane / 4 + $r * 8;
                    unsafe {
                        output.add(row * 256 + col).cast::<Float2>().write(Float2 {
                            values: [output_tile[j][$r * 2], output_tile[j][$r * 2 + 1]],
                        });
                    }
                }};
            }
            store!(0);
            store!(1);
            j += 1;
        }
    }
}
