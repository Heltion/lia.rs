use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

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
    pub unsafe fn apply(logits: &[f32], penalties: &[f32], output: *mut f32) {
        let i = thread::index_1d().get();
        if i < logits.len() {
            unsafe {
                *output.add(i) = logits[i] - penalties[i % penalties.len()];
            }
        }
    }
}
