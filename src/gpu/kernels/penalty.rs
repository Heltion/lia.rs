use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[kernel]
    pub unsafe fn remember(token: usize, strength: f32, penalties: *mut f32) {
        if thread::index_1d().get() == 0 {
            unsafe {
                *penalties.add(token) = strength;
            }
        }
    }

    #[kernel]
    pub unsafe fn apply(
        logits: &[f32],
        penalties: &[f32],
        prefix: &[u32],
        count: usize,
        causal: bool,
        strength: f32,
        output: *mut f32,
    ) {
        let i = thread::index_1d().get();
        if i < logits.len() {
            let token = i % penalties.len();
            let end = if causal {
                (i / penalties.len() + 1).min(count)
            } else {
                count
            };
            let mut penalty = penalties[token];
            for j in 0..end {
                if prefix[j] as usize == token {
                    penalty = strength;
                }
            }
            unsafe {
                *output.add(i) = logits[i] - penalty;
            }
        }
    }
}
