//! Float32 token-to-mel convolution, relative positions and attention operators.
use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_module;
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub fn codec_columns(
        frames: usize,
        kernel: usize,
        left: usize,
        repeat: usize,
        input: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = out.get_mut(idx) {
            let row = i / (512 * kernel);
            let channel = i % (512 * kernel) / kernel;
            let sample = row / (frames * repeat);
            let source = (row % (frames * repeat)) as isize + (i % kernel) as isize - left as isize;
            *value = if source < 0 || source >= (frames * repeat) as isize {
                0.0
            } else {
                input[(sample * frames + source as usize / repeat) * 512 + channel]
            };
        }
    }
    #[kernel]
    pub unsafe fn codec_affine(
        count: usize,
        dim: usize,
        activation: usize,
        bias: &[f32],
        data: *mut f32,
    ) {
        let i = thread::index_1d().get();
        if i < count {
            let x = unsafe { *data.add(i) } + bias[i % dim];
            let y = if activation == 1 {
                x / (1.0 + (-x).exp())
            } else if activation == 2 {
                x.max(0.0) + 0.01 * x.min(0.0)
            } else if activation == 3 {
                0.5 * x * (1.0 + (0.797_884_6 * x * (1.0 + 0.044_715 * x * x)).tanh())
            } else {
                x
            };
            unsafe {
                *data.add(i) = y;
            }
        }
    }
    #[kernel]
    pub unsafe fn codec_scale(count: usize, scale: f32, data: *mut f32) {
        let i = thread::index_1d().get();
        if i < count {
            unsafe {
                *data.add(i) *= scale;
            }
        }
    }
    #[kernel]
    pub fn codec_positions(frames: usize, mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = out.get_mut(idx) {
            let position = (frames - 1) as isize - (i / 512) as isize;
            let frequency = (-((i % 512 / 2 * 2) as f32) * 0.017988946).exp();
            let angle = position as f32 * frequency;
            *value = if i % 2 == 0 { angle.sin() } else { angle.cos() };
        }
    }
    #[kernel]
    pub unsafe fn codec_relative_attention(
        frames: usize,
        qkv: &[f32],
        pos: &[f32],
        u: &[f32],
        v: &[f32],
        out: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        if group < frames * 8 {
            let row = group / 8;
            let head = group % 8;
            let channel = head * 64 + lane;
            let q = [qkv[row * 1536 + channel], qkv[row * 1536 + channel + 32]];
            let mut result = [0.0f32; 2];
            let mut maximum = f32::NEG_INFINITY;
            let mut total = 0.0;
            let mut token = 0;
            while token < frames {
                let key = token * 1536 + 512 + channel;
                let p = (frames - 1 + token - row) * 512 + channel;
                let mut dot = (q[0] + u[channel]) * qkv[key]
                    + (q[1] + u[channel + 32]) * qkv[key + 32]
                    + (q[0] + v[channel]) * pos[p]
                    + (q[1] + v[channel + 32]) * pos[p + 32];
                let mut delta = 16;
                while delta > 0 {
                    dot += warp::shuffle_down_f32_sync(0xffff_ffff, dot, delta);
                    delta /= 2;
                }
                let score = warp::shuffle_f32_sync(0xffff_ffff, dot, 0) * 0.125;
                let next = maximum.max(score);
                let keep = cuda_device::float::ex2_approx_f32(
                    (maximum - next) * core::f32::consts::LOG2_E,
                );
                let probability =
                    cuda_device::float::ex2_approx_f32((score - next) * core::f32::consts::LOG2_E);
                total = total * keep + probability;
                result[0] =
                    qkv[token * 1536 + 1024 + channel].mul_add(probability, result[0] * keep);
                result[1] =
                    qkv[token * 1536 + 1024 + channel + 32].mul_add(probability, result[1] * keep);
                maximum = next;
                token += 1;
            }
            unsafe {
                *out.add(row * 512 + channel) = result[0] / total;
                *out.add(row * 512 + channel + 32) = result[1] / total;
            }
        }
    }
}
