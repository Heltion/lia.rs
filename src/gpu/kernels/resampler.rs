//! Adaptive 2D sine/cosine keys and 128-dimensional perceiver cross attention.
use super::minicpm_bf16::rounded;
use cuda_device::{kernel, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub unsafe fn resampler_positions<const BF16: bool>(
        width: usize,
        input: &[f32],
        out: *mut f32,
    ) {
        let i = thread::index_1d().get();
        if i < input.len() {
            let row = i / 4096;
            let dim = i % 4096;
            let coordinate = if dim < 2048 { row % width } else { row / width };
            let angle = coordinate as f32 * (1.0 / 10000.0f32.powf((dim % 1024) as f32 / 1024.0));
            let position = if dim % 2048 < 1024 {
                angle.sin()
            } else {
                angle.cos()
            };
            let value = if BF16 {
                rounded(input[i] + rounded(position))
            } else {
                input[i] + position
            };
            unsafe {
                *out.add(i) = value;
            }
        }
    }

    #[kernel]
    pub unsafe fn resampler_attention(
        queries: usize,
        keys: usize,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        if group < queries * 32 {
            let head = group % 32;
            let mut query = [0.0f32; 4];
            let mut values = [0.0f32; 4];
            let mut d = 0;
            #[unroll]
            while d < 4 {
                query[d] = unsafe { *q.get_unchecked(group * 128 + lane + d * 32) };
                d += 1;
            }
            let mut maximum = f32::NEG_INFINITY;
            let mut total = 0.0f32;
            let mut token = 0;
            while token < keys {
                let at = token * 4096 + head * 128 + lane;
                let mut dot = 0.0f32;
                d = 0;
                #[unroll]
                while d < 4 {
                    dot = query[d].mul_add(unsafe { *k.get_unchecked(at + d * 32) }, dot);
                    d += 1;
                }
                let mut delta = 16;
                while delta > 0 {
                    dot += warp::shuffle_down_f32_sync(0xffff_ffff, dot, delta);
                    delta /= 2;
                }
                let score = warp::shuffle_f32_sync(0xffff_ffff, dot, 0) * 0.088_388_346;
                let next = maximum.max(score);
                let alpha = cuda_device::float::ex2_approx_f32(
                    (maximum - next) * core::f32::consts::LOG2_E,
                );
                let probability =
                    cuda_device::float::ex2_approx_f32((score - next) * core::f32::consts::LOG2_E);
                total = total * alpha + probability;
                d = 0;
                #[unroll]
                while d < 4 {
                    values[d] = (unsafe { *v.get_unchecked(at + d * 32) })
                        .mul_add(probability, values[d] * alpha);
                    d += 1;
                }
                maximum = next;
                token += 1;
            }
            d = 0;
            #[unroll]
            while d < 4 {
                unsafe {
                    *out.add(group * 128 + lane + d * 32) = values[d] / total;
                }
                d += 1;
            }
        }
    }
}
