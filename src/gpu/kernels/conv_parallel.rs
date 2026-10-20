//! Independent causal four-tap windows with an immutable starting history.
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[kernel]
    pub unsafe fn conv_silu_parallel(
        channels: usize,
        input: &[f32],
        weights: &[f32],
        initial: &[f32],
        history: *mut f32,
        captured: *mut f32,
        out: *mut f32,
    ) {
        let index = thread::index_1d().get();
        let batch = input.len() / channels;
        let channel = index % channels;
        let start = index / channels * 4;
        if start >= batch {
            return;
        }
        let base = channel * 3;
        // SAFETY: complete channel rows and a disjoint three-value snapshot.
        // Later time tiles read their preceding inputs, never the mutable final
        // history. Consequently no thread waits for another time tile.
        unsafe {
            let mut a = if start == 0 {
                initial[base]
            } else {
                input[(start - 3) * channels + channel]
            };
            let mut b = if start == 0 {
                initial[base + 1]
            } else {
                input[(start - 2) * channels + channel]
            };
            let mut c = if start == 0 {
                initial[base + 2]
            } else {
                input[(start - 1) * channels + channel]
            };
            let w0 = weights[channel * 4];
            let w1 = weights[channel * 4 + 1];
            let w2 = weights[channel * 4 + 2];
            let w3 = weights[channel * 4 + 3];
            let mut step = 0;
            #[unroll]
            while step < 4 {
                let t = start + step;
                if t < batch {
                    let x = input[t * channels + channel];
                    let value = a * w0 + b * w1 + c * w2 + x * w3;
                    if !captured.is_null() {
                        *captured.add(t * channels * 3 + base) = b;
                        *captured.add(t * channels * 3 + base + 1) = c;
                        *captured.add(t * channels * 3 + base + 2) = x;
                    }
                    *out.add(t * channels + channel) = value / (1.0 + (-value).exp());
                    if t + 1 == batch {
                        *history.add(base) = b;
                        *history.add(base + 1) = c;
                        *history.add(base + 2) = x;
                    }
                    a = b;
                    b = c;
                    c = x;
                }
                step += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};

    #[test]
    fn time_tiles_preserve_output_final_and_captured_state_bits() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = crate::gpu::kernels::load(&context)?;
        for (channels, batch) in [
            (1usize, 1usize),
            (17, 2),
            (17, 3),
            (33, 4),
            (257, 17),
            (8192, 65),
            (257, 513),
        ] {
            let input = (0..channels * batch)
                .map(|i| ((i * 17 % 137) as f32 - 68.0) / 31.0)
                .collect::<Vec<_>>();
            let weights = (0..channels * 4)
                .map(|i| ((i * 23 % 41) as f32 - 20.0) / 67.0)
                .collect::<Vec<_>>();
            let initial = (0..channels * 3)
                .map(|i| ((i * 31 % 43) as f32 - 21.0) / 53.0)
                .collect::<Vec<_>>();
            let mut state = initial.clone();
            state.extend([12345.0; 17]);
            let x = DeviceBuffer::from_host(&stream, &input)?;
            let w = DeviceBuffer::from_host(&stream, &weights)?;
            let snapshot = DeviceBuffer::from_host(&stream, &initial)?;
            let a = DeviceBuffer::from_host(&stream, &state)?;
            let b = DeviceBuffer::from_host(&stream, &state)?;
            let out_a = DeviceBuffer::from_host(&stream, &vec![12345.0f32; input.len() + 17])?;
            let out_b = DeviceBuffer::from_host(&stream, &vec![12345.0f32; input.len() + 17])?;
            let cap_a = DeviceBuffer::from_host(&stream, &vec![12345.0f32; input.len() * 3 + 17])?;
            let cap_b = DeviceBuffer::from_host(&stream, &vec![12345.0f32; input.len() * 3 + 17])?;
            // SAFETY: complete independent channel windows, an immutable history
            // snapshot and guarded, disjoint outputs for both implementations.
            unsafe {
                module.convolution.conv_silu(
                    &stream,
                    LaunchConfig::for_num_elems(channels as u32),
                    channels,
                    &x,
                    &w,
                    a.cu_deviceptr() as *mut f32,
                    cap_a.cu_deviceptr() as *mut f32,
                    out_a.cu_deviceptr() as *mut f32,
                )?;
                module.conv_parallel.conv_silu_parallel(
                    &stream,
                    LaunchConfig::for_num_elems((batch.div_ceil(4) * channels) as u32),
                    channels,
                    &x,
                    &w,
                    &snapshot,
                    b.cu_deviceptr() as *mut f32,
                    cap_b.cu_deviceptr() as *mut f32,
                    out_b.cu_deviceptr() as *mut f32,
                )?;
            }
            let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
            assert_eq!(bits(a.to_host_vec(&stream)?), bits(b.to_host_vec(&stream)?));
            assert_eq!(
                bits(cap_a.to_host_vec(&stream)?),
                bits(cap_b.to_host_vec(&stream)?)
            );
            let actual = out_b.to_host_vec(&stream)?;
            assert_eq!(bits(out_a.to_host_vec(&stream)?), bits(actual.clone()));
            assert_eq!(&actual[input.len()..], &[12345.0; 17]);
            for t in 0..batch {
                for c in 0..channels {
                    let mut sum = 0.0f64;
                    for tap in 0..4 {
                        let value = if t + tap < 3 {
                            initial[c * 3 + t + tap]
                        } else {
                            input[(t + tap - 3) * channels + c]
                        };
                        sum += value as f64 * weights[c * 4 + tap] as f64;
                    }
                    let expected = sum / (1.0 + (-sum).exp());
                    assert!(
                        (actual[t * channels + c] as f64 - expected).abs()
                            <= expected.abs() * 0.000002 + 0.000002
                    );
                }
            }
            assert_eq!(snapshot.to_host_vec(&stream)?, initial);
        }
        Ok(())
    }
}
