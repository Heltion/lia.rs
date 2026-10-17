use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub unsafe fn delta_net<const PRECOMPUTED: bool>(
        dim: usize,
        k_heads: usize,
        v_heads: usize,
        qk: &[f32],
        conv: &[f32],
        alpha: &[f32],
        beta: &[f32],
        dt: &[f32],
        a: &[f32],
        state: *mut f32,
        captured: *mut f32,
        out: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let row = tid / 32;
        let lane = tid % 32;
        let head = row / dim;
        let value = row % dim;
        if head < v_heads {
            let mut t = 0;
            while t < alpha.len() / v_heads {
                let kh = head % k_heads;
                let qoff = t * 2 * k_heads * dim + kh * dim;
                let koff = t * 2 * k_heads * dim + (k_heads + kh) * dim;
                let v = conv
                    [t * (2 * k_heads + v_heads) * dim + 2 * k_heads * dim + head * dim + value];
                let b = if PRECOMPUTED {
                    beta[t * v_heads + head]
                } else {
                    1.0 / (1.0 + (-beta[t * v_heads + head]).exp())
                };
                let decay = if PRECOMPUTED {
                    alpha[t * v_heads + head]
                } else {
                    let x = alpha[t * v_heads + head] + dt[head];
                    let softplus = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
                    (a[head] * softplus).exp()
                };
                let mut dot = 0.0f32;
                let mut j = lane;
                while j < dim {
                    unsafe {
                        dot += *state.add(row * dim + j) * qk[koff + j];
                    }
                    j += 32;
                }
                let mut delta = 16;
                while delta > 0 {
                    dot += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, dot, delta);
                    delta /= 2;
                }
                dot = cuda_device::warp::shuffle_f32_sync(0xffff_ffff, dot, 0);
                let correction = (v - decay * dot) * b;
                let mut result = 0.0f32;
                j = lane;
                while j < dim {
                    unsafe {
                        let s = decay * *state.add(row * dim + j) + qk[koff + j] * correction;
                        *state.add(row * dim + j) = s;
                        if !captured.is_null() {
                            *captured.add(t * v_heads * dim * dim + row * dim + j) = s;
                        }
                        result += s * qk[qoff + j];
                    }
                    j += 32;
                }
                delta = 16;
                while delta > 0 {
                    result += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, result, delta);
                    delta /= 2;
                }
                if lane == 0 {
                    unsafe {
                        *out.add(t * v_heads * dim + row) = result / (dim as f32).sqrt();
                    }
                }
                t += 1;
            }
        }
    }
}
