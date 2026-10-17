use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub unsafe fn conv_silu(
        channels: usize,
        input: &[f32],
        weights: &[f32],
        history: *mut f32,
        captured: *mut f32,
        out: *mut f32,
    ) {
        let i = thread::index_1d().get();
        if i < channels {
            unsafe {
                let base = i * 3;
                let mut t = 0;
                while t < input.len() / channels {
                    let a = *history.add(base);
                    let b = *history.add(base + 1);
                    let c = *history.add(base + 2);
                    let x = input[t * channels + i];
                    let value = a * weights[i * 4]
                        + b * weights[i * 4 + 1]
                        + c * weights[i * 4 + 2]
                        + x * weights[i * 4 + 3];
                    *history.add(base) = b;
                    *history.add(base + 1) = c;
                    *history.add(base + 2) = x;
                    if !captured.is_null() {
                        *captured.add(t * channels * 3 + base) = b;
                        *captured.add(t * channels * 3 + base + 1) = c;
                        *captured.add(t * channels * 3 + base + 2) = x;
                    }
                    *out.add(t * channels + i) = value / (1.0 + (-value).exp());
                    t += 1;
                }
            }
        }
    }

    #[kernel]
    pub unsafe fn l2_norm(
        dim: usize,
        groups: usize,
        per_token: usize,
        stride: usize,
        eps: f32,
        x: &[f32],
        out: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        if group < groups {
            let mut sum = 0.0f32;
            let mut j = lane;
            while j < dim {
                let v = x[(group / per_token) * stride + (group % per_token) * dim + j];
                sum += v * v;
                j += 32;
            }
            let mut delta = 16;
            while delta > 0 {
                sum += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum, delta);
                delta /= 2;
            }
            sum = cuda_device::warp::shuffle_f32_sync(0xffff_ffff, sum, 0);
            let scale = 1.0 / (sum + eps).sqrt();
            j = lane;
            while j < dim {
                unsafe {
                    *out.add(group * dim + j) =
                        x[(group / per_token) * stride + (group % per_token) * dim + j] * scale;
                }
                j += 32;
            }
        }
    }

    #[kernel]
    pub unsafe fn gdn_gates(
        heads: usize,
        alpha: &[f32],
        beta: &[f32],
        dt: &[f32],
        a: &[f32],
        decay: *mut f32,
        bnorm: *mut f32,
    ) {
        let i = thread::index_1d().get();
        if i < alpha.len() {
            let h = i % heads;
            let x = alpha[i] + dt[h];
            let softplus = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
            unsafe {
                *decay.add(i) = (a[h] * softplus).exp();
                *bnorm.add(i) = 1.0 / (1.0 + (-beta[i]).exp());
            }
        }
    }
}
