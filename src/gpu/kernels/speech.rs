//! Speech decoder width-64 rotary/cache attention and unit-norm conditioning.
use cuda_device::{kernel, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub unsafe fn speech_rope_kv(
        position: usize,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        rotated: *mut f32,
        kc: *mut f32,
        vc: *mut f32,
    ) {
        let i = thread::index_1d().get();
        if i < q.len() {
            let d = i % 64;
            let pair = if d < 32 { i + 32 } else { i - 32 };
            let angle =
                (position + i / 768) as f32 * (1.0 / 10000.0f32.powf((d % 32) as f32 / 32.0));
            let cosine = angle.cos();
            let sine = if d < 32 { -angle.sin() } else { angle.sin() };
            unsafe {
                *rotated.add(i) = q[i] * cosine + q[pair] * sine;
                *kc.add(position * 768 + i) = k[i] * cosine + k[pair] * sine;
                *vc.add(position * 768 + i) = v[i];
            }
        }
    }
    #[kernel]
    pub unsafe fn speech_attention(
        position: usize,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        if group < q.len() / 64 {
            let row = group / 12;
            let head = group % 12;
            let query = [q[group * 64 + lane], q[group * 64 + lane + 32]];
            let mut accum = [0.0f32; 2];
            let mut maximum = f32::NEG_INFINITY;
            let mut total = 0.0;
            let mut token = 0;
            while token <= position + row {
                let at = token * 768 + head * 64 + lane;
                let mut dot = query[0] * k[at] + query[1] * k[at + 32];
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
                let p =
                    cuda_device::float::ex2_approx_f32((score - next) * core::f32::consts::LOG2_E);
                total = total * keep + p;
                accum[0] = v[at].mul_add(p, accum[0] * keep);
                accum[1] = v[at + 32].mul_add(p, accum[1] * keep);
                maximum = next;
                token += 1;
            }
            unsafe {
                *out.add(group * 64 + lane) = accum[0] / total;
                *out.add(group * 64 + lane + 32) = accum[1] / total;
            }
        }
    }
    #[kernel]
    pub unsafe fn speech_unit_norm(input: &[f32], out: *mut f32) {
        let tid = thread::index_1d().get();
        let row = tid / 32;
        let lane = tid % 32;
        if row < input.len() / 768 {
            let mut sum = 0.0;
            let mut d = lane;
            while d < 768 {
                let x = input[row * 768 + d];
                sum += x * x;
                d += 32;
            }
            let mut delta = 16;
            while delta > 0 {
                sum += warp::shuffle_down_f32_sync(0xffff_ffff, sum, delta);
                delta /= 2;
            }
            let norm = warp::shuffle_f32_sync(0xffff_ffff, sum, 0)
                .sqrt()
                .max(1e-12);
            d = lane;
            while d < 768 {
                unsafe {
                    *out.add(row * 768 + d) = input[row * 768 + d] / norm;
                }
                d += 32;
            }
        }
    }
}
