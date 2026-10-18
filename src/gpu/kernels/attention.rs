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
        group_offset: usize,
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
                    *out.add((group + group_offset) * DIM + lane) = o0
                        / sum
                        / (1.0 + (-qgate[(group + group_offset) * DIM * 2 + DIM + lane]).exp());
                }
                if lane + 32 < DIM {
                    *out.add((group + group_offset) * DIM + lane + 32) = o1
                        / sum
                        / (1.0
                            + (-qgate[(group + group_offset) * DIM * 2 + DIM + lane + 32]).exp());
                }
                if lane + 2 * 32 < DIM {
                    *out.add((group + group_offset) * DIM + lane + 2 * 32) = o2
                        / sum
                        / (1.0
                            + (-qgate[(group + group_offset) * DIM * 2 + DIM + lane + 2 * 32])
                                .exp());
                }
                if lane + 3 * 32 < DIM {
                    *out.add((group + group_offset) * DIM + lane + 3 * 32) = o3
                        / sum
                        / (1.0
                            + (-qgate[(group + group_offset) * DIM * 2 + DIM + lane + 3 * 32])
                                .exp());
                }
                if lane + 4 * 32 < DIM {
                    *out.add((group + group_offset) * DIM + lane + 4 * 32) = o4
                        / sum
                        / (1.0
                            + (-qgate[(group + group_offset) * DIM * 2 + DIM + lane + 4 * 32])
                                .exp());
                }
                if lane + 5 * 32 < DIM {
                    *out.add((group + group_offset) * DIM + lane + 5 * 32) = o5
                        / sum
                        / (1.0
                            + (-qgate[(group + group_offset) * DIM * 2 + DIM + lane + 5 * 32])
                                .exp());
                }
                if lane + 6 * 32 < DIM {
                    *out.add((group + group_offset) * DIM + lane + 6 * 32) = o6
                        / sum
                        / (1.0
                            + (-qgate[(group + group_offset) * DIM * 2 + DIM + lane + 6 * 32])
                                .exp());
                }
                if lane + 7 * 32 < DIM {
                    *out.add((group + group_offset) * DIM + lane + 7 * 32) = o7
                        / sum
                        / (1.0
                            + (-qgate[(group + group_offset) * DIM * 2 + DIM + lane + 7 * 32])
                                .exp());
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

#[cfg(test)]
mod tests {
    use crate::gpu::{kernels, tests::close};
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};

    #[test]
    fn attention() -> Result<()> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let module = kernels::load(&ctx)?;
        let dim = 8;
        let qh = 4;
        let kvh = 2;
        let tokens = 3;
        let capacity = 5;
        let q: Vec<f32> = (0..qh * dim)
            .map(|i| ((i * 5 % 19) as f32 - 9.) * 0.1)
            .collect();
        let k: Vec<f32> = (0..tokens * kvh * dim)
            .map(|i| ((i * 7 % 23) as f32 - 11.) * 0.1)
            .collect();
        let v: Vec<f32> = (0..tokens * kvh * dim)
            .map(|i| ((i * 11 % 17) as f32 - 8.) * 0.1)
            .collect();
        let gate: Vec<f32> = (0..qh * dim * 2)
            .map(|i| ((i * 3 % 13) as f32 - 6.) * 0.1)
            .collect();
        let dq = DeviceBuffer::from_host(&stream, &q)?;
        let dk = DeviceBuffer::from_host(&stream, &k)?;
        let dv = DeviceBuffer::from_host(&stream, &v)?;
        let dg = DeviceBuffer::from_host(&stream, &gate)?;
        let scores = DeviceBuffer::<f32>::zeroed(&stream, qh * capacity)?;
        let mut out = DeviceBuffer::<f32>::zeroed(&stream, qh * dim)?;
        // SAFETY: all Q/K/V and score extents match their explicit dimensions; each
        // phase has disjoint output ownership and uses the same ordered stream.
        unsafe {
            module.attention.attn_scores(
                &stream,
                LaunchConfig::for_num_elems((qh * tokens * 32) as u32),
                dim,
                qh,
                kvh,
                tokens,
                capacity,
                &dq,
                &dk,
                scores.cu_deviceptr() as *mut f32,
            )?;
            module.attention.softmax(
                &stream,
                LaunchConfig::for_num_elems((qh * 32) as u32),
                qh,
                tokens,
                capacity,
                scores.cu_deviceptr() as *mut f32,
            )?;
            module.attention.attn_values(
                &stream,
                LaunchConfig::for_num_elems((qh * dim) as u32),
                dim,
                qh,
                kvh,
                tokens,
                capacity,
                &scores,
                &dv,
                &dg,
                &mut out,
            )?;
        }
        let mut expected = vec![0f64; qh * dim];
        for h in 0..qh {
            let mut probs: Vec<f64> = (0..tokens)
                .map(|t| {
                    (0..dim)
                        .map(|j| {
                            q[h * dim + j] as f64
                                * k[t * kvh * dim + (h / (qh / kvh)) * dim + j] as f64
                        })
                        .sum::<f64>()
                        / (dim as f64).sqrt()
                })
                .collect();
            let max = probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            probs.iter_mut().for_each(|p| *p = (*p - max).exp());
            let sum: f64 = probs.iter().sum();
            probs.iter_mut().for_each(|p| *p /= sum);
            for j in 0..dim {
                expected[h * dim + j] = (0..tokens)
                    .map(|t| probs[t] * v[t * kvh * dim + (h / (qh / kvh)) * dim + j] as f64)
                    .sum::<f64>()
                    / (1. + (-gate[h * dim * 2 + dim + j] as f64).exp());
            }
        }
        close(&out.to_host_vec(&stream)?, &expected, "grouped attention")?;
        println!("GPU grouped causal attention, softmax and sigmoid gating match f64 reference");
        let parts = tokens.div_ceil(2);
        let cap_parts = 4;
        let partial = DeviceBuffer::<f32>::zeroed(&stream, qh * cap_parts * (dim + 2))?;
        let fused = DeviceBuffer::<f32>::zeroed(&stream, qh * dim)?;
        // SAFETY: two-token context partitions cover the same validated attention
        // problem; merge outputs own one disjoint vector per query head.
        unsafe {
            module.flash.flash_partial::<8, 2>(
                &stream,
                LaunchConfig::for_num_elems((qh * parts * 32) as u32),
                qh,
                kvh,
                tokens,
                parts,
                cap_parts,
                &dq,
                &dk,
                &dv,
                partial.cu_deviceptr() as *mut f32,
            )?;
            module.attention.flash_merge::<8>(
                &stream,
                LaunchConfig::for_num_elems((qh * 32) as u32),
                qh,
                parts,
                cap_parts,
                0,
                &partial,
                &dg,
                fused.cu_deviceptr() as *mut f32,
            )?;
        }
        close(
            &fused.to_host_vec(&stream)?,
            &expected,
            "partitioned online attention",
        )?;
        println!("GPU partitioned online attention matches the independent f64 reference");

        Ok(())
    }
}
