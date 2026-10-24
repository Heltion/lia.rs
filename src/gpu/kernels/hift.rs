//! HiFT temporal column gathering and pointwise Float32 activations.
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub fn hift_columns(
        channels: usize,
        kernel: usize,
        left: usize,
        dilation: usize,
        stride: usize,
        input: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let row = i / (channels * kernel);
            let channel = i / kernel % channels;
            let position = row * stride + (i % kernel) * dilation;
            *value = if position < left || position - left >= input.len() / channels {
                0.0
            } else {
                input[(position - left) * channels + channel]
            };
        }
    }
    #[kernel]
    pub unsafe fn hift_affine(
        channels: usize,
        mode: usize,
        count: usize,
        bias: &[f32],
        output: *mut f32,
    ) {
        let i = thread::index_1d().get();
        if i < count {
            let value = unsafe { *output.add(i) } + bias[i % channels];
            let result = match mode {
                1 => {
                    if value < 0.0 {
                        value.exp() - 1.0
                    } else {
                        value
                    }
                }
                2 => value.abs(),
                4 => value.tanh(),
                3 => {
                    if value < 0.0 {
                        value * 0.1
                    } else {
                        value
                    }
                }
                _ => value,
            };
            unsafe {
                *output.add(i) = result;
            }
        }
    }
}
