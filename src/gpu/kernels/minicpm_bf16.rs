//! BF16 activation boundaries for the MiniCPM vision encoder.
use cuda_device::{convert, kernel, thread, warp};
use cuda_host::cuda_module;

#[inline(always)]
pub(super) fn rounded(value: f32) -> f32 {
    convert::cvt_f32x2_bf16x2(convert::cvt_bf16x2_f32(value, value)).0
}

#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[kernel]
    pub unsafe fn round(count: usize, values: *mut f32) {
        let i = thread::index_1d().get() * 2;
        if i < count {
            let a = unsafe { *values.add(i) };
            let b = if i + 1 < count {
                unsafe { *values.add(i + 1) }
            } else {
                0.0
            };
            let (a, b) = convert::cvt_f32x2_bf16x2(convert::cvt_bf16x2_f32(a, b));
            unsafe {
                *values.add(i) = a;
                if i + 1 < count {
                    *values.add(i + 1) = b;
                }
            }
        }
    }

    #[kernel]
    pub unsafe fn affine<const GELU: bool>(
        count: usize,
        dim: usize,
        bias: &[f32],
        values: *mut f32,
    ) {
        let i = thread::index_1d().get();
        if i < count {
            let x = rounded(unsafe { *values.add(i) } + bias[i % dim]);
            let y = if GELU {
                rounded(0.5 * x * (1.0 + (0.797_884_6 * x * (1.0 + 0.044_715 * x * x)).tanh()))
            } else {
                x
            };
            unsafe {
                *values.add(i) = y;
            }
        }
    }

    #[inline(always)]
    fn sum(mut value: f32) -> f32 {
        let mut offset = 16;
        while offset > 0 {
            value += warp::shuffle_down_f32_sync(0xffff_ffff, value, offset);
            offset /= 2;
        }
        warp::shuffle_f32_sync(0xffff_ffff, value, 0)
    }

    /// At most sixteen keys, covering the four-token insert windows and tiny images.
    #[kernel]
    pub unsafe fn attention<const WINDOW: usize>(tokens: usize, qkv: &[f32], output: *mut f32) {
        let tid = thread::index_1d().get();
        let query = tid / 512;
        let head = tid / 32 % 16;
        let lane = tid % 32;
        if query >= tokens {
            return;
        }
        let count = if WINDOW == 4 { 4 } else { tokens };
        let first = if WINDOW == 4 { query / 4 * 4 } else { 0 };
        let mut scores = [0.0f32; 16];
        let mut maximum = f32::NEG_INFINITY;
        let mut key = 0;
        while key < count {
            let mut dot = 0.0;
            let mut c = lane;
            while c < 72 {
                dot += qkv[query * 3456 + head * 72 + c]
                    * qkv[(first + key) * 3456 + 1152 + head * 72 + c];
                c += 32;
            }
            scores[key] = sum(dot) / 72.0f32.sqrt();
            maximum = maximum.max(scores[key]);
            key += 1;
        }
        let mut weights = [0.0f32; 16];
        let mut denominator = 0.0;
        key = 0;
        while key < count {
            let weight = (scores[key] - maximum).exp();
            denominator += weight;
            weights[key] = rounded(weight);
            key += 1;
        }
        let mut c = lane;
        while c < 72 {
            let mut value = 0.0;
            key = 0;
            while key < count {
                value += weights[key] * qkv[(first + key) * 3456 + 2304 + head * 72 + c];
                key += 1;
            }
            unsafe {
                *output.add(query * 1152 + head * 72 + c) = rounded(value / denominator);
            }
            c += 32;
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};

    #[test]
    fn round_matches_ieee_ties_to_even_and_preserves_tail() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = super::super::load(&context)?;
        let mut source: Vec<f32> = (0..73)
            .map(|i| f32::from_bits(0x3f80_0000 + i * 0x4000))
            .collect();
        source.extend([-0.0, -1.00390625, 0.00000001, -13.03125]);
        let count = source.len();
        source.extend([123.4567, -321.7654]);
        let values = DeviceBuffer::from_host(&stream, &source)?;
        // SAFETY: one guarded writer per pair; count excludes the two sentinel values.
        unsafe {
            module.minicpm_bf16.round(
                &stream,
                LaunchConfig::for_num_elems(count.div_ceil(2) as u32),
                count,
                values.cu_deviceptr() as *mut f32,
            )?;
        }
        let actual = values.to_host_vec(&stream)?;
        for (&input, &output) in source[..count].iter().zip(&actual) {
            let bits = input.to_bits();
            let expected = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
            assert_eq!(output.to_bits(), expected);
        }
        assert_eq!(&actual[count..], &source[count..]);
        Ok(())
    }

    #[test]
    fn attention_preserves_uniform_bf16_value_means() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = super::super::load(&context)?;
        for (tokens, window) in [
            (3usize, false),
            (16, false),
            (17, false),
            (65, false),
            (130, false),
            (257, false),
            (8, true),
            (16, true),
        ] {
            let mut qkv = vec![0.0f32; tokens * 3456];
            for token in 0..tokens {
                for head in 0..16 {
                    for channel in 0..72 {
                        qkv[token * 3456 + 2304 + head * 72 + channel] =
                            ((token % 7) as i32 + head as i32 - 8 + (channel % 3) as i32) as f32
                                * 0.125;
                    }
                }
            }
            let input = DeviceBuffer::from_host(&stream, &qkv)?;
            let output = DeviceBuffer::<f32>::zeroed(&stream, tokens * 1152)?;
            // SAFETY: exact QKV/output extents, complete heads, and checked small windows.
            unsafe {
                let launch = LaunchConfig::for_num_elems((tokens * 512) as u32);
                if window {
                    module.minicpm_bf16.attention::<4>(
                        &stream,
                        launch,
                        tokens,
                        &input,
                        output.cu_deviceptr() as *mut f32,
                    )?;
                } else if tokens <= 16 {
                    module.minicpm_bf16.attention::<0>(
                        &stream,
                        launch,
                        tokens,
                        &input,
                        output.cu_deviceptr() as *mut f32,
                    )?;
                } else {
                    module.vision_flash.tiled_attention::<32, true>(
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
                module.minicpm_bf16.round(
                    &stream,
                    LaunchConfig::for_num_elems((output.len() / 2) as u32),
                    output.len(),
                    output.cu_deviceptr() as *mut f32,
                )?;
            }
            let actual = output.to_host_vec(&stream)?;
            for query in 0..tokens {
                let first = if window { query / 4 * 4 } else { 0 };
                let count = if window { 4 } else { tokens };
                for head in 0..16 {
                    for channel in 0..72 {
                        let mean = (first..first + count)
                            .map(|key| f64::from(qkv[key * 3456 + 2304 + head * 72 + channel]))
                            .sum::<f64>()
                            / count as f64;
                        let bits = (mean as f32).to_bits();
                        let expected = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
                        assert_eq!(
                            actual[query * 1152 + head * 72 + channel].to_bits(),
                            expected,
                            "BF16 mean tokens={tokens} query={query} head={head} channel={channel}"
                        );
                    }
                }
            }
        }
        Ok(())
    }
}
