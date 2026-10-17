use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;
    /// A 128-wide recurrent row has four elements per lane. Keep them live
    /// across the token batch instead of rereading the matrix for every step.
    #[kernel]
    pub unsafe fn delta_net_128(
        k_heads: usize,
        v_heads: usize,
        qk: &[f32],
        conv: &[f32],
        decay: &[f32],
        beta: &[f32],
        state: *mut f32,
        captured: *mut f32,
        out: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let row = tid / 32;
        let lane = tid % 32;
        let head = row / 128;
        let value = row % 128;
        if head < v_heads {
            let base = row * 128 + lane;
            let (mut s0, mut s1, mut s2, mut s3) = unsafe {
                (
                    *state.add(base),
                    *state.add(base + 32),
                    *state.add(base + 64),
                    *state.add(base + 96),
                )
            };
            let mut t = 0;
            while t < decay.len() / v_heads {
                let qoff = t * 2 * k_heads * 128 + (head % k_heads) * 128 + lane;
                let koff = qoff + k_heads * 128;
                let k0 = qk[koff];
                let k1 = qk[koff + 32];
                let k2 = qk[koff + 64];
                let k3 = qk[koff + 96];
                let mut dot = s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3;
                let mut delta = 16;
                while delta > 0 {
                    dot += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, dot, delta);
                    delta /= 2;
                }
                dot = cuda_device::warp::shuffle_f32_sync(0xffff_ffff, dot, 0);
                let d = decay[t * v_heads + head];
                let v = conv
                    [t * (2 * k_heads + v_heads) * 128 + 2 * k_heads * 128 + head * 128 + value];
                let correction = (v - d * dot) * beta[t * v_heads + head];
                s0 = d * s0 + k0 * correction;
                s1 = d * s1 + k1 * correction;
                s2 = d * s2 + k2 * correction;
                s3 = d * s3 + k3 * correction;
                if !captured.is_null() {
                    unsafe {
                        let off = t * v_heads * 128 * 128 + base;
                        *captured.add(off) = s0;
                        *captured.add(off + 32) = s1;
                        *captured.add(off + 64) = s2;
                        *captured.add(off + 96) = s3;
                    }
                }
                let mut result =
                    s0 * qk[qoff] + s1 * qk[qoff + 32] + s2 * qk[qoff + 64] + s3 * qk[qoff + 96];
                delta = 16;
                while delta > 0 {
                    result += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, result, delta);
                    delta /= 2;
                }
                if lane == 0 {
                    unsafe {
                        *out.add(t * v_heads * 128 + row) = result / 128.0f32.sqrt();
                    }
                }
                t += 1;
            }
            unsafe {
                *state.add(base) = s0;
                *state.add(base + 32) = s1;
                *state.add(base + 64) = s2;
                *state.add(base + 96) = s3;
            }
        }
    }
}
