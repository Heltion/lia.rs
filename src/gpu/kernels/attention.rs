use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;
    // Softmax differences are nonpositive. CUDA's hardware exponential has
    // sufficient precision here; keep recurrence gates on the standard path.
    #[inline(always)]
    fn softmax_exp(x: f32) -> f32 {
        cuda_device::float::ex2_approx_f32(x * core::f32::consts::LOG2_E)
    }

    #[kernel]
    pub unsafe fn flash_merge<const DIM: usize>(
        groups: usize,
        parts: usize,
        cap_parts: usize,
        partial: &[f32],
        qgate: &[f32],
        out: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        if group < groups {
            let mut max = f32::NEG_INFINITY;
            let mut p = 0;
            while p < parts {
                max = max.max(partial[(group * cap_parts + p) * (DIM + 2) + DIM]);
                p += 1;
            }
            let mut sum = 0.0f32;
            let mut o0 = 0.0f32;
            let mut o1 = 0.0f32;
            let mut o2 = 0.0f32;
            let mut o3 = 0.0f32;
            let mut o4 = 0.0f32;
            let mut o5 = 0.0f32;
            let mut o6 = 0.0f32;
            let mut o7 = 0.0f32;
            p = 0;
            while p < parts {
                let base = (group * cap_parts + p) * (DIM + 2);
                let weight = softmax_exp(partial[base + DIM] - max);
                sum += weight * partial[base + DIM + 1];
                if lane < DIM {
                    o0 += weight * partial[base + lane];
                }
                if lane + 32 < DIM {
                    o1 += weight * partial[base + lane + 32];
                }
                if lane + 2 * 32 < DIM {
                    o2 += weight * partial[base + lane + 2 * 32];
                }
                if lane + 3 * 32 < DIM {
                    o3 += weight * partial[base + lane + 3 * 32];
                }
                if lane + 4 * 32 < DIM {
                    o4 += weight * partial[base + lane + 4 * 32];
                }
                if lane + 5 * 32 < DIM {
                    o5 += weight * partial[base + lane + 5 * 32];
                }
                if lane + 6 * 32 < DIM {
                    o6 += weight * partial[base + lane + 6 * 32];
                }
                if lane + 7 * 32 < DIM {
                    o7 += weight * partial[base + lane + 7 * 32];
                }
                p += 1;
            }
            unsafe {
                if lane < DIM {
                    *out.add(group * DIM + lane) =
                        o0 / sum / (1.0 + (-qgate[group * DIM * 2 + DIM + lane]).exp());
                }
                if lane + 32 < DIM {
                    *out.add(group * DIM + lane + 32) =
                        o1 / sum / (1.0 + (-qgate[group * DIM * 2 + DIM + lane + 32]).exp());
                }
                if lane + 2 * 32 < DIM {
                    *out.add(group * DIM + lane + 2 * 32) =
                        o2 / sum / (1.0 + (-qgate[group * DIM * 2 + DIM + lane + 2 * 32]).exp());
                }
                if lane + 3 * 32 < DIM {
                    *out.add(group * DIM + lane + 3 * 32) =
                        o3 / sum / (1.0 + (-qgate[group * DIM * 2 + DIM + lane + 3 * 32]).exp());
                }
                if lane + 4 * 32 < DIM {
                    *out.add(group * DIM + lane + 4 * 32) =
                        o4 / sum / (1.0 + (-qgate[group * DIM * 2 + DIM + lane + 4 * 32]).exp());
                }
                if lane + 5 * 32 < DIM {
                    *out.add(group * DIM + lane + 5 * 32) =
                        o5 / sum / (1.0 + (-qgate[group * DIM * 2 + DIM + lane + 5 * 32]).exp());
                }
                if lane + 6 * 32 < DIM {
                    *out.add(group * DIM + lane + 6 * 32) =
                        o6 / sum / (1.0 + (-qgate[group * DIM * 2 + DIM + lane + 6 * 32]).exp());
                }
                if lane + 7 * 32 < DIM {
                    *out.add(group * DIM + lane + 7 * 32) =
                        o7 / sum / (1.0 + (-qgate[group * DIM * 2 + DIM + lane + 7 * 32]).exp());
                }
            }
        }
    }

    #[kernel]
    pub unsafe fn attn_scores(
        dim: usize,
        q_heads: usize,
        kv_heads: usize,
        tokens: usize,
        capacity: usize,
        q: &[f32],
        k: &[f32],
        scores: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let row = tid / 32;
        let lane = tid % 32;
        let group = row / tokens;
        let head = group % q_heads;
        let query = group / q_heads;
        let t = row % tokens;
        let batch = q.len() / (q_heads * dim);
        if query < batch {
            if t > tokens - batch + query {
                if lane == 0 {
                    unsafe {
                        *scores.add(group * capacity + t) = f32::NEG_INFINITY;
                    }
                }
                return;
            }
            let kh = head / (q_heads / kv_heads);
            let mut sum = 0.0f32;
            let mut j = lane;
            while j < dim {
                sum += q[group * dim + j] * k[t * kv_heads * dim + kh * dim + j];
                j += 32;
            }
            let mut delta = 16;
            while delta > 0 {
                sum += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum, delta);
                delta /= 2;
            }
            if lane == 0 {
                unsafe {
                    *scores.add(group * capacity + t) = sum / (dim as f32).sqrt();
                }
            }
        }
    }

    #[kernel]
    pub unsafe fn softmax(heads: usize, tokens: usize, capacity: usize, scores: *mut f32) {
        let tid = thread::index_1d().get();
        let head = tid / 32;
        let lane = tid % 32;
        if head < heads {
            let mut max = f32::NEG_INFINITY;
            let mut j = lane;
            while j < tokens {
                unsafe {
                    max = max.max(*scores.add(head * capacity + j));
                }
                j += 32;
            }
            let mut delta = 16;
            while delta > 0 {
                max = max.max(cuda_device::warp::shuffle_down_f32_sync(
                    0xffff_ffff,
                    max,
                    delta,
                ));
                delta /= 2;
            }
            max = cuda_device::warp::shuffle_f32_sync(0xffff_ffff, max, 0);
            let mut sum = 0.0f32;
            j = lane;
            while j < tokens {
                unsafe {
                    sum += (*scores.add(head * capacity + j) - max).exp();
                }
                j += 32;
            }
            delta = 16;
            while delta > 0 {
                sum += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum, delta);
                delta /= 2;
            }
            sum = cuda_device::warp::shuffle_f32_sync(0xffff_ffff, sum, 0);
            j = lane;
            while j < tokens {
                unsafe {
                    let p = scores.add(head * capacity + j);
                    *p = (*p - max).exp() / sum;
                }
                j += 32;
            }
        }
    }

    #[kernel]
    pub fn attn_values(
        dim: usize,
        q_heads: usize,
        kv_heads: usize,
        tokens: usize,
        capacity: usize,
        probs: &[f32],
        v: &[f32],
        qgate: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = out.get_mut(idx) {
            let head = i / dim;
            let lane = i % dim;
            let kh = (head % q_heads) / (q_heads / kv_heads);
            let mut sum = 0.0f32;
            let mut t = 0;
            while t < tokens {
                sum += probs[head * capacity + t] * v[t * kv_heads * dim + kh * dim + lane];
                t += 1;
            }
            *value = sum / (1.0 + (-qgate[head * dim * 2 + dim + lane]).exp());
        }
    }
}
