//! Causal 128-dimensional GQA with online softmax and FP16 KV storage.
use cuda_device::{kernel, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[kernel]
    pub unsafe fn split_qkv(input: &[f32], q: *mut f32, k: *mut f32, v: *mut f32) {
        let i = thread::index_1d().get();
        if i < input.len() {
            let token = i / 6144;
            let row = i % 6144;
            unsafe {
                if row < 4096 {
                    *q.add(token * 4096 + row) = input[i];
                } else if row < 5120 {
                    *k.add(token * 1024 + row - 4096) = input[i];
                } else {
                    *v.add(token * 1024 + row - 5120) = input[i];
                }
            }
        }
    }

    #[kernel]
    pub unsafe fn dense_store_kv(offset: usize, k: &[f32], v: &[f32], kc: *mut u16, vc: *mut u16) {
        let i = thread::index_1d().get();
        if i < k.len() {
            unsafe {
                *kc.add(offset + i) = (k[i] as f16).to_bits();
                *vc.add(offset + i) = (v[i] as f16).to_bits();
            }
        }
    }

    #[kernel]
    pub unsafe fn dense_attention128(
        queries: usize,
        query_heads: usize,
        kv_heads: usize,
        position: usize,
        parts: usize,
        span: usize,
        q: &[f32],
        k: &[u16],
        v: &[u16],
        output: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let group = tid / 32 / parts;
        let part = tid / 32 % parts;
        let lane = tid % 32;
        if group < queries * query_heads {
            let row = group / query_heads;
            let head = group % query_heads;
            let kv_head = head / (query_heads / kv_heads);
            let mut query = [0.0f32; 4];
            let mut accum = [0.0f32; 4];
            let mut d = 0;
            #[unroll]
            while d < 4 {
                query[d] = unsafe { *q.get_unchecked(group * 128 + lane + d * 32) };
                d += 1;
            }
            let mut maximum = f32::NEG_INFINITY;
            let mut denominator = 0.0f32;
            let mut token = part * span;
            let end = if span == 0 {
                position + row + 1
            } else {
                ((part + 1) * span).min(position + row + 1)
            };
            while token < end {
                let at = (token * kv_heads + kv_head) * 128 + lane;
                let mut dot = 0.0f32;
                d = 0;
                #[unroll]
                while d < 4 {
                    let key = f16::from_bits(unsafe { *k.get_unchecked(at + d * 32) }) as f32;
                    dot = query[d].mul_add(key, dot);
                    d += 1;
                }
                let mut delta = 16;
                while delta > 0 {
                    dot += warp::shuffle_down_f32_sync(0xffff_ffff, dot, delta);
                    delta /= 2;
                }
                let score = warp::shuffle_f32_sync(0xffff_ffff, dot, 0) * 0.08838834764831845;
                let next = maximum.max(score);
                let alpha = cuda_device::float::ex2_approx_f32(
                    (maximum - next) * core::f32::consts::LOG2_E,
                );
                let probability =
                    cuda_device::float::ex2_approx_f32((score - next) * core::f32::consts::LOG2_E);
                denominator = denominator * alpha + probability;
                d = 0;
                #[unroll]
                while d < 4 {
                    let value = f16::from_bits(unsafe { *v.get_unchecked(at + d * 32) }) as f32;
                    accum[d] = (accum[d] * alpha).mul_add(1.0, value * probability);
                    d += 1;
                }
                maximum = next;
                token += 1;
            }
            d = 0;
            #[unroll]
            while d < 4 {
                unsafe {
                    if span == 0 {
                        *output.add(group * 128 + lane + d * 32) = accum[d] / denominator;
                    } else {
                        *output.add((group * parts + part) * 130 + lane + d * 32) = accum[d];
                    }
                }
                d += 1;
            }
            if span != 0 && lane == 0 {
                unsafe {
                    *output.add((group * parts + part) * 130 + 128) = maximum;
                    *output.add((group * parts + part) * 130 + 129) = denominator;
                }
            }
        }
    }

    /// Merge unnormalized partition values with their own softmax maxima.
    #[kernel]
    pub unsafe fn dense_attention_merge(
        heads: usize,
        parts: usize,
        partial: &[f32],
        output: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let head = tid / 32;
        let lane = tid % 32;
        if head < heads {
            let mut maximum = f32::NEG_INFINITY;
            let mut part = 0;
            while part < parts {
                maximum = maximum.max(partial[(head * parts + part) * 130 + 128]);
                part += 1;
            }
            let mut values = [0.0f32; 4];
            let mut total = 0.0f32;
            part = 0;
            while part < parts {
                let at = (head * parts + part) * 130;
                let mass = partial[at + 129];
                let scale = if mass > 0.0 {
                    cuda_device::float::ex2_approx_f32(
                        (partial[at + 128] - maximum) * core::f32::consts::LOG2_E,
                    )
                } else {
                    0.0
                };
                total += scale * mass;
                let mut d = 0;
                #[unroll]
                while d < 4 {
                    values[d] += partial[at + lane + d * 32] * scale;
                    d += 1;
                }
                part += 1;
            }
            let mut d = 0;
            #[unroll]
            while d < 4 {
                unsafe {
                    *output.add(head * 128 + lane + d * 32) = values[d] / total;
                }
                d += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::gpu::kernels;
    use anyhow::{Result, ensure};
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};

    #[test]
    fn partitioned_decode_matches_serial_and_f64_at_context_boundaries() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        for tokens in [
            1usize, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1344, 8192,
        ] {
            for span in [32usize, 64, 128, 256] {
                let q: Vec<f32> = (0..32 * 128)
                    .map(|i| ((i * 17 % 37) as f32 - 18.) * 0.17)
                    .collect();
                let k: Vec<f16> = (0..tokens * 8 * 128)
                    .map(|i| (((i * 23 % 43) as f32 - 21.) * 0.15) as f16)
                    .collect();
                let v: Vec<f16> = (0..k.len())
                    .map(|i| (((i * 31 % 47) as f32 - 23.) * 0.12) as f16)
                    .collect();
                let dq = DeviceBuffer::from_host(&stream, &q)?;
                let dk = DeviceBuffer::from_host(
                    &stream,
                    &k.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                )?;
                let dv = DeviceBuffer::from_host(
                    &stream,
                    &v.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                )?;
                let parts = tokens.div_ceil(span);
                let guard = 12345.0f32;
                let serial = DeviceBuffer::from_host(&stream, &vec![guard; 4096 + 17])?;
                let merged = DeviceBuffer::from_host(&stream, &vec![guard; 4096 + 17])?;
                let partial =
                    DeviceBuffer::from_host(&stream, &vec![guard; 32 * parts * 130 + 17])?;
                // SAFETY: all complete 32/8-head arrays and guarded output extents
                // match the serial and partitioned launches, including the last tile.
                unsafe {
                    module.dense_attention.dense_attention128(
                        &stream,
                        LaunchConfig::for_num_elems(1024),
                        1,
                        32,
                        8,
                        tokens - 1,
                        1,
                        0,
                        &dq,
                        &dk,
                        &dv,
                        serial.cu_deviceptr() as *mut f32,
                    )?;
                    module.dense_attention.dense_attention128(
                        &stream,
                        LaunchConfig::for_num_elems((1024 * parts) as u32),
                        1,
                        32,
                        8,
                        tokens - 1,
                        parts,
                        span,
                        &dq,
                        &dk,
                        &dv,
                        partial.cu_deviceptr() as *mut f32,
                    )?;
                    module.dense_attention.dense_attention_merge(
                        &stream,
                        LaunchConfig::for_num_elems(1024),
                        32,
                        parts,
                        &partial,
                        merged.cu_deviceptr() as *mut f32,
                    )?;
                }
                let a = serial.to_host_vec(&stream)?;
                let b = merged.to_host_vec(&stream)?;
                let p = partial.to_host_vec(&stream)?;
                ensure!(
                    a[4096..]
                        .iter()
                        .chain(&b[4096..])
                        .chain(&p[32 * parts * 130..])
                        .all(|x| *x == guard),
                    "attention output guard overwritten"
                );
                let maximum = a[..4096]
                    .iter()
                    .zip(&b)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                ensure!(
                    maximum < 0.00005 && b[..4096].iter().all(|x| x.is_finite()),
                    "partitioned attention differs: {maximum}"
                );
                let mut oracle_error = 0.0f64;
                for head in [0, 3, 4, 31] {
                    let mut scores: Vec<f64> = (0..tokens)
                        .map(|t| {
                            (0..128)
                                .map(|d| {
                                    f64::from(q[head * 128 + d])
                                        * f64::from(k[(t * 8 + head / 4) * 128 + d] as f32)
                                })
                                .sum::<f64>()
                                / 128.0f64.sqrt()
                        })
                        .collect();
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    for score in &mut scores {
                        *score = (*score - max).exp();
                    }
                    let mass: f64 = scores.iter().sum();
                    for d in 0..128 {
                        let expected: f64 = scores
                            .iter()
                            .enumerate()
                            .map(|(t, w)| w * f64::from(v[(t * 8 + head / 4) * 128 + d] as f32))
                            .sum::<f64>()
                            / mass;
                        oracle_error =
                            oracle_error.max((f64::from(b[head * 128 + d]) - expected).abs());
                    }
                }
                println!(
                    "attention span={span} tokens={tokens}: serial_error={maximum}, f64_error={oracle_error}"
                );
                ensure!(oracle_error < 0.00005, "attention independent f64 mismatch");
            }
        }
        Ok(())
    }
}
