//! Stable GPU grouping and deterministic reduction of routed expert choices.
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[kernel]
    pub unsafe fn group(batch: usize, selected: &[u32], counts: *mut u32, choices: *mut u32) {
        let expert = thread::index_1d().get();
        if expert < 256 {
            let mut count = 0;
            let mut choice = 0;
            while choice < batch * 8 {
                if selected[choice] as usize == expert {
                    // The router selects each expert at most once per token.
                    if count < batch {
                        unsafe {
                            *choices.add(expert * batch + count) = choice as u32;
                        }
                    }
                    count += 1;
                }
                choice += 1;
            }
            unsafe {
                *counts.add(expert) = count.min(batch) as u32;
            }
        }
    }

    #[kernel]
    pub unsafe fn descriptors(counts: &[u32], tiles: *mut u32) {
        if thread::index_1d().get() == 0 {
            let mut expert = 0;
            let mut tile = 0;
            while expert < 256 {
                let mut start = 0;
                while start < counts[expert] {
                    unsafe {
                        *tiles.add(1 + tile * 2) = expert as u32;
                        *tiles.add(2 + tile * 2) = start;
                    }
                    tile += 1;
                    start += 16;
                }
                expert += 1;
            }
            unsafe {
                *tiles = tile as u32;
            }
        }
    }

    #[kernel]
    pub unsafe fn merge(
        batch: usize,
        hidden: usize,
        choices: &[f32],
        probabilities: &[f32],
        output: *mut f32,
    ) {
        let i = thread::index_1d().get();
        if i < batch * hidden {
            let token = i / hidden;
            let row = i % hidden;
            let mut sum = 0.0;
            let mut slot = 0;
            while slot < 8 {
                sum += choices[(token * 8 + slot) * hidden + row] * probabilities[token * 8 + slot];
                slot += 1;
            }
            unsafe {
                *output.add(i) = sum;
            }
        }
    }
}
