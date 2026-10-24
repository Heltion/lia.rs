//! S3 rotary width-64 attention, depthwise temporal memory and finite scalar codes.
use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_module;
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub fn s3_rotary(q: &[f32], k: &[f32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let frame = i / 2560;
            let col = i % 1280;
            let d = col % 64;
            let source = if i % 2560 < 1280 { q } else { k };
            let angle = frame as f32 / 10000.0f32.powf((d % 32) as f32 / 32.);
            let pair = if d < 32 {
                -source[frame * 1280 + col + 32]
            } else {
                source[frame * 1280 + col - 32]
            };
            *value = source[frame * 1280 + col] * angle.cos() + pair * angle.sin();
        }
    }
    #[kernel]
    pub unsafe fn s3_attention(frames: usize, qk: &[f32], v: &[f32], out: *mut f32) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        if group < frames * 20 {
            let row = group / 20;
            let channel = group % 20 * 64 + lane;
            // Upstream scales Q and K independently by head_dim^(-1/4).
            let scale = 0.35355338f32;
            let query = [
                qk[row * 2560 + channel] * scale,
                qk[row * 2560 + channel + 32] * scale,
            ];
            let mut maximum = f32::NEG_INFINITY;
            let mut total = 0.;
            let mut accum = [0.0f32; 2];
            for token in 0..frames {
                let key = token * 2560 + 1280 + channel;
                let mut dot = query[0] * (qk[key] * scale) + query[1] * (qk[key + 32] * scale);
                let mut step = 16;
                while step > 0 {
                    dot += warp::shuffle_down_f32_sync(0xffff_ffff, dot, step);
                    step /= 2;
                }
                let score = warp::shuffle_f32_sync(0xffff_ffff, dot, 0);
                let next = maximum.max(score);
                let keep = cuda_device::float::ex2_approx_f32(
                    (maximum - next) * core::f32::consts::LOG2_E,
                );
                let probability =
                    cuda_device::float::ex2_approx_f32((score - next) * core::f32::consts::LOG2_E);
                total = total * keep + probability;
                accum[0] = v[token * 1280 + channel].mul_add(probability, accum[0] * keep);
                accum[1] = v[token * 1280 + channel + 32].mul_add(probability, accum[1] * keep);
                maximum = next;
            }
            unsafe {
                *out.add(row * 1280 + channel) = accum[0] / total;
                *out.add(row * 1280 + channel + 32) = accum[1] / total;
            }
        }
    }
    #[kernel]
    pub fn s3_memory(frames: usize, input: &[f32], weight: &[f32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let frame = i / 1280;
            let channel = i % 1280;
            let mut sum = 0.0f32;
            for tap in 0..31 {
                let source = frame as isize + tap as isize - 15;
                if source >= 0 && source < frames as isize {
                    sum = weight[channel * 31 + tap]
                        .mul_add(input[source as usize * 1280 + channel], sum);
                }
            }
            *value = sum + input[i];
        }
    }
    #[kernel]
    pub fn s3_codes(input: &[f32], mut output: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let mut code = 0;
            let mut power = 1;
            for d in 0..8 {
                let bounded = input[i * 8 + d].tanh() * 0.9990000128746033;
                let level = if bounded > 0.5 {
                    2
                } else if bounded < -0.5 {
                    0
                } else {
                    1
                };
                code += level * power;
                power *= 3;
            }
            *value = code;
        }
    }
}
