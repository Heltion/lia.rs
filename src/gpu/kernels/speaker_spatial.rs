//! CAM++ frequency/time convolution gathering and final channel flattening.
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub fn speaker_columns(
        bins: usize,
        channels: usize,
        kf: usize,
        kt: usize,
        stride: usize,
        pf: usize,
        pt: usize,
        out_bins: usize,
        input: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let row = i / (channels * kf * kt);
            let col = i % (channels * kf * kt);
            let channel = col / (kf * kt);
            let frequency = row % out_bins * stride + (col / kt) % kf;
            let time = row / out_bins + col % kt;
            *value = if frequency < pf
                || frequency - pf >= bins
                || time < pt
                || time - pt >= input.len() / (bins * channels)
            {
                0.
            } else {
                input[((time - pt) * bins + frequency - pf) * channels + channel]
            };
        }
    }
    #[kernel]
    pub fn speaker_flatten(
        bins: usize,
        channels: usize,
        input: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let time = i / (bins * channels);
            let channel = i / bins % channels;
            let bin = i % bins;
            *value = input[(time * bins + bin) * channels + channel];
        }
    }
}
