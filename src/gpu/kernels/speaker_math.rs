//! CAM++ channel affine, normalization, concatenation and gated context operations.
use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_module;
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[inline(always)]
    fn sum(mut x: f32) -> f32 {
        let mut delta = 16;
        while delta > 0 {
            x += warp::shuffle_down_f32_sync(0xffff_ffff, x, delta);
            delta /= 2;
        }
        warp::shuffle_f32_sync(0xffff_ffff, x, 0)
    }
    #[kernel]
    pub unsafe fn speaker_affine(
        channels: usize,
        mode: usize,
        count: usize,
        bias: &[f32],
        out: *mut f32,
    ) {
        let i = thread::index_1d().get();
        if i < count {
            let x = unsafe { *out.add(i) } + bias[i % channels];
            unsafe {
                *out.add(i) = if mode == 1 {
                    x.max(0.)
                } else if mode == 2 {
                    1. / (1. + (-x).exp())
                } else {
                    x
                };
            }
        }
    }
    #[kernel]
    pub fn speaker_norm(
        relu: bool,
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        mean: &[f32],
        variance: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let c = i % weight.len();
            let scale = weight[c] / (variance[c] + 1e-5).sqrt();
            let shift = bias[c] - mean[c] * scale;
            let x = input[i].mul_add(scale, shift);
            *value = if relu { x.max(0.) } else { x };
        }
    }
    #[kernel]
    pub fn speaker_relu(input: &[f32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            *value = input[i].max(0.);
        }
    }
    #[kernel]
    pub fn speaker_concat(
        left: usize,
        right: usize,
        a: &[f32],
        b: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let frame = i / (left + right);
            let c = i % (left + right);
            *value = if c < left {
                a[frame * left + c]
            } else {
                b[frame * right + c - left]
            };
        }
    }
    #[kernel]
    pub unsafe fn speaker_context(frames: usize, channels: usize, input: &[f32], out: *mut f32) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        if group < frames.div_ceil(100) * channels {
            let channel = group % channels;
            let segment = group / channels;
            let mut total = 0.;
            let mut frame = lane;
            while frame < frames {
                total += input[frame * channels + channel];
                frame += 32;
            }
            let mean = sum(total) / frames as f32;
            let mut part = 0.;
            frame = segment * 100 + lane;
            while frame < ((segment + 1) * 100).min(frames) {
                part += input[frame * channels + channel];
                frame += 32;
            }
            let local = sum(part) / ((frames - segment * 100).min(100)) as f32;
            if lane == 0 {
                unsafe {
                    *out.add(group) = mean + local;
                }
            }
        }
    }
    #[kernel]
    pub fn speaker_gate(
        channels: usize,
        input: &[f32],
        gate: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            *value = input[i] * gate[(i / channels / 100) * channels + i % channels];
        }
    }
    #[kernel]
    pub unsafe fn speaker_stats(frames: usize, channels: usize, input: &[f32], out: *mut f32) {
        let tid = thread::index_1d().get();
        let channel = tid / 32;
        let lane = tid % 32;
        if channel < channels {
            let mut total = 0.;
            let mut frame = lane;
            while frame < frames {
                total += input[frame * channels + channel];
                frame += 32;
            }
            let mean = sum(total) / frames as f32;
            let mut squares = 0.;
            frame = lane;
            while frame < frames {
                let d = input[frame * channels + channel] - mean;
                squares = d.mul_add(d, squares);
                frame += 32;
            }
            let std = (sum(squares) / (frames - 1) as f32).sqrt();
            if lane == 0 {
                unsafe {
                    *out.add(channel) = mean;
                    *out.add(channels + channel) = std;
                }
            }
        }
    }
}
