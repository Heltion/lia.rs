//! Head normalization and noncausal attention, isolated across CFG samples.
use cuda_device::{kernel, thread, warp};
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
    pub unsafe fn cfm_qk_norm(
        rows: usize,
        qw: &[f32],
        qb: &[f32],
        kw: &[f32],
        kb: &[f32],
        qkv: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        if group < rows * 16 {
            let row = group / 16;
            let part = group % 16 / 8;
            let head = group % 8;
            let at = row * 1536 + part * 512 + head * 64 + lane;
            let a = unsafe { *qkv.add(at) };
            let b = unsafe { *qkv.add(at + 32) };
            let mean = sum(a + b) / 64.0;
            let scale =
                1.0 / (sum((a - mean) * (a - mean) + (b - mean) * (b - mean)) / 64.0 + 1e-5).sqrt();
            let (weight, bias) = if part == 0 { (qw, qb) } else { (kw, kb) };
            unsafe {
                *qkv.add(at) = (a - mean) * scale * weight[lane] + bias[lane];
                *qkv.add(at + 32) = (b - mean) * scale * weight[lane + 32] + bias[lane + 32];
            }
        }
    }
    #[kernel]
    pub unsafe fn cfm_attention(frames: usize, qkv: &[f32], out: *mut f32) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        if group < qkv.len() / 1536 * 8 {
            let row = group / 8;
            let head = group % 8;
            let sample = row / frames;
            let channel = head * 64 + lane;
            let query = [qkv[row * 1536 + channel], qkv[row * 1536 + channel + 32]];
            let mut result = [0.0f32; 2];
            let mut total = 0.0;
            let mut maximum = f32::NEG_INFINITY;
            let mut token = 0;
            while token < frames {
                let at = (sample * frames + token) * 1536 + 512 + channel;
                let score = sum(query[0] * qkv[at] + query[1] * qkv[at + 32]) * 0.125;
                let next = maximum.max(score);
                let keep = cuda_device::float::ex2_approx_f32(
                    (maximum - next) * core::f32::consts::LOG2_E,
                );
                let p =
                    cuda_device::float::ex2_approx_f32((score - next) * core::f32::consts::LOG2_E);
                total = total * keep + p;
                result[0] = qkv[at + 512].mul_add(p, result[0] * keep);
                result[1] = qkv[at + 544].mul_add(p, result[1] * keep);
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
