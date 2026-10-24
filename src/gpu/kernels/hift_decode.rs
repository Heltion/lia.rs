//! HiFT transpose convolution gathering, Snake and reflection boundary operations.
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub fn hift_transpose_columns(
        channels: usize,
        kernel: usize,
        padding: usize,
        stride: usize,
        input: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let frame = i / (channels * kernel);
            let channel = i / kernel % channels;
            let k = i % kernel;
            let position = frame + padding;
            *value = if position < k
                || !(position - k).is_multiple_of(stride)
                || (position - k) / stride >= input.len() / channels
            {
                0.0
            } else {
                input[((position - k) / stride) * channels + channel]
            };
        }
    }
    #[kernel]
    pub fn hift_snake(input: &[f32], alpha: &[f32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let a = alpha[i % alpha.len()];
            let sine = (a * input[i]).sin();
            *value = input[i] + (1.0 / (a + 1e-9)) * (sine * sine);
        }
    }
    #[kernel]
    pub fn hift_leaky(slope: f32, input: &[f32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let x = input[i];
            *value = if x >= 0.0 { x } else { x * slope };
        }
    }
    #[kernel]
    pub fn hift_reflect_left(channels: usize, input: &[f32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let frame = i / channels;
            let at = if frame == 0 { 1 } else { frame - 1 };
            *value = input[at * channels + i % channels];
        }
    }
    #[kernel]
    pub fn hift_mean(a: &[f32], b: &[f32], c: &[f32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            *value = ((a[i] + b[i]) + c[i]) / 3.0;
        }
    }
}
