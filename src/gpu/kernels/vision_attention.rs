//! Noncausal, online-softmax attention for 16 vision heads of width 72.
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[kernel]
    pub unsafe fn attention<const WINDOW: usize>(tokens: usize, qkv: &[f32], output: *mut f32) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        if group < tokens * 16 {
            let query = group / 16;
            let head = group % 16;
            let base = query * 3456 + head * 72 + lane;
            let q0 = qkv[base];
            let q1 = qkv[base + 32];
            let q2 = if lane < 8 { qkv[base + 64] } else { 0.0 };
            let mut o0 = 0.0;
            let mut o1 = 0.0;
            let mut o2 = 0.0;
            let mut max = f32::NEG_INFINITY;
            let mut sum = 0.0;
            let mut key = if WINDOW == 0 {
                0
            } else {
                query / WINDOW * WINDOW
            };
            let end = if WINDOW == 0 {
                tokens
            } else {
                (key + WINDOW).min(tokens)
            };
            while key < end {
                let kb = key * 3456 + 1152 + head * 72 + lane;
                let mut score = q0 * qkv[kb] + q1 * qkv[kb + 32];
                if lane < 8 {
                    score += q2 * qkv[kb + 64];
                }
                let mut step = 16;
                while step > 0 {
                    score += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, score, step);
                    step /= 2;
                }
                score = cuda_device::warp::shuffle_f32_sync(0xffff_ffff, score, 0) / 72.0f32.sqrt();
                let next = max.max(score);
                let keep = (max - next).exp();
                let probability = (score - next).exp();
                sum = sum * keep + probability;
                let vb = key * 3456 + 2304 + head * 72 + lane;
                o0 = o0 * keep + probability * qkv[vb];
                o1 = o1 * keep + probability * qkv[vb + 32];
                if lane < 8 {
                    o2 = o2 * keep + probability * qkv[vb + 64];
                }
                max = next;
                key += 1;
            }
            let dst = query * 1152 + head * 72 + lane;
            unsafe {
                *output.add(dst) = o0 / sum;
                *output.add(dst + 32) = o1 / sum;
                if lane < 8 {
                    *output.add(dst + 64) = o2 / sum;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};

    #[test]
    fn four_patch_windows_match_independent_softmax() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = super::super::load(&context)?;
        let tokens = 12usize;
        let qkv: Vec<f32> = (0..tokens * 3456)
            .map(|i| ((i * 73 % 257) as f32 - 128.0) / 271.0)
            .collect();
        let input = DeviceBuffer::from_host(&stream, &qkv)?;
        let output = DeviceBuffer::<f32>::zeroed(&stream, tokens * 1152)?;
        // SAFETY: three complete four-patch windows,16 heads of72 channels.
        unsafe {
            module.vision_attention.attention::<4>(
                &stream,
                LaunchConfig::for_num_elems((tokens * 16 * 32) as u32),
                tokens,
                &input,
                output.cu_deviceptr() as *mut f32,
            )?;
        }
        let actual = output.to_host_vec(&stream)?;
        for query in 0..tokens {
            for head in 0..16 {
                let keys = query / 4 * 4..query / 4 * 4 + 4;
                let scores: Vec<f64> = keys
                    .clone()
                    .map(|key| {
                        (0..72)
                            .map(|c| {
                                qkv[query * 3456 + head * 72 + c] as f64
                                    * qkv[key * 3456 + 1152 + head * 72 + c] as f64
                            })
                            .sum::<f64>()
                            / 72.0f64.sqrt()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let probabilities: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum = probabilities.iter().sum::<f64>();
                for channel in 0..72 {
                    let expected = keys
                        .clone()
                        .zip(&probabilities)
                        .map(|(key, p)| p * qkv[key * 3456 + 2304 + head * 72 + channel] as f64)
                        .sum::<f64>()
                        / sum;
                    assert!(
                        (actual[query * 1152 + head * 72 + channel] as f64 - expected).abs() < 1e-6
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn all_patches_participate_in_stable_attention() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = super::super::load(&context)?;
        for (tokens, scale) in [(1usize, 1.0), (7, 0.2), (20, 8.0), (65, 0.2)] {
            let qkv: Vec<f32> = (0..tokens * 3456)
                .map(|i| ((i * 73 % 257) as f32 - 128.0) * scale / 37.0)
                .collect();
            let input = DeviceBuffer::from_host(&stream, &qkv)?;
            let output = DeviceBuffer::<f32>::zeroed(&stream, tokens * 1152)?;
            let reference = DeviceBuffer::<f32>::zeroed(&stream, tokens * 1152)?;
            // SAFETY: complete QKV rows, padded query tails, and CTA sizes
            // matching each tile. Both kernels retain the same FP16 residuals.
            unsafe {
                module.vision_flash.tiled_attention::<16, false>(
                    &stream,
                    LaunchConfig {
                        grid_dim: ((tokens.div_ceil(16) * 16) as u32, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    tokens,
                    &input,
                    reference.cu_deviceptr() as *mut f32,
                )?;
                module.vision_flash.tiled_attention::<32, false>(
                    &stream,
                    LaunchConfig {
                        grid_dim: ((tokens.div_ceil(32) * 16) as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    tokens,
                    &input,
                    output.cu_deviceptr() as *mut f32,
                )?;
            }
            let reference = reference.to_host_vec(&stream)?;
            let actual = output.to_host_vec(&stream)?;
            let scalar = DeviceBuffer::<f32>::zeroed(&stream, tokens * 1152)?;
            // SAFETY: the scalar comparison uses the same complete QKV rows.
            unsafe {
                module.vision_attention.attention::<0>(
                    &stream,
                    LaunchConfig::for_num_elems((tokens * 16 * 32) as u32),
                    tokens,
                    &input,
                    scalar.cu_deviceptr() as *mut f32,
                )?;
            }
            let scalar = scalar.to_host_vec(&stream)?;
            let mut max_error = 0.0f64;
            let mut scalar_error = 0.0f64;
            for query in 0..tokens {
                for head in 0..16 {
                    let scores: Vec<f64> = (0..tokens)
                        .map(|key| {
                            (0..72)
                                .map(|d| {
                                    qkv[query * 3456 + head * 72 + d] as f64
                                        * qkv[key * 3456 + 1152 + head * 72 + d] as f64
                                })
                                .sum::<f64>()
                                / 72.0f64.sqrt()
                        })
                        .collect();
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let probabilities: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                    let sum: f64 = probabilities.iter().sum();
                    for d in 0..72 {
                        let expected: f64 = probabilities
                            .iter()
                            .enumerate()
                            .map(|(key, p)| p * qkv[key * 3456 + 2304 + head * 72 + d] as f64)
                            .sum::<f64>()
                            / sum;
                        let value = actual[query * 1152 + head * 72 + d] as f64;
                        max_error = max_error.max((value - expected).abs());
                        scalar_error = scalar_error
                            .max((scalar[query * 1152 + head * 72 + d] as f64 - expected).abs());
                        // The extreme fixture produces logits above 1000;
                        // F32 MMA accumulation order can move nearly tied softmax scores.
                        // Keep a separate absolute bound for that stress case.
                        let tolerance = if scale > 1.0 {
                            0.02
                        } else {
                            3e-5 * (1.0 + expected.abs())
                        };
                        let prior = reference[query * 1152 + head * 72 + d] as f64;
                        assert!((prior - expected).abs() < tolerance, "16-query reference");
                        assert!((value - prior).abs() < tolerance, "query tile agreement");
                        assert!(
                            (value - expected).abs() < tolerance,
                            "tokens={tokens} query={query} head={head} d={d}: {value} != {expected}"
                        );
                    }
                }
            }
            println!(
                "tokens={tokens} scale={scale} tiled_max_abs_error={max_error} scalar_max_abs_error={scalar_error}"
            );
        }
        Ok(())
    }
}
