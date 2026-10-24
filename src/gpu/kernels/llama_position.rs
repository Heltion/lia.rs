//! Prefill projections share rotary coefficients and write paged KV directly.
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[repr(C, align(8))]
    #[derive(Clone, Copy)]
    struct Pair {
        values: [f32; 2],
    }

    #[kernel]
    pub unsafe fn concatenate(q: &[u8], k: &[u8], v: &[u8], output: *mut u32) {
        let i = thread::index_1d().get();
        let q_words = q.len() / 4;
        let k_words = k.len() / 4;
        let (source, index) = if i < q_words {
            (q, i)
        } else if i < q_words + k_words {
            (k, i - q_words)
        } else {
            (v, i - q_words - k_words)
        };
        if index < source.len() / 4 {
            unsafe {
                *output.add(i) = source.as_ptr().cast::<u32>().add(index).read();
            }
        }
    }

    #[kernel]
    pub unsafe fn coefficients(rows: usize, base: f32, positions: &[u32], output: *mut f32) {
        let i = thread::index_1d().get();
        if i < rows * 64 {
            // Preserve the official reciprocal-of-positive-power rounding order.
            let angle = positions[i / 64] as f32 * (1.0 / base.powf(2.0 * (i % 64) as f32 / 128.0));
            unsafe {
                *output.add(i * 2) = angle.cos();
                *output.add(i * 2 + 1) = angle.sin();
            }
        }
    }

    #[kernel]
    pub unsafe fn rotary_cache(
        rows: usize,
        stride: usize,
        positions: &[u32],
        sequences: &[u32],
        pages: &[u32],
        coefficients: &[f32],
        input: &[f32],
        query: *mut f32,
        keys: *mut u16,
        values: *mut u16,
    ) {
        let i = thread::index_1d().get();
        if i < rows * 1024 {
            let row = i / 1024;
            let head = i % 1024 / 64;
            let d = i % 64;
            let [cosine, sine] = unsafe {
                coefficients
                    .as_ptr()
                    .add(row * 128 + d * 2)
                    .cast::<Pair>()
                    .read()
                    .values
            };
            let offset = row * 2560 + head * 128;
            let left = input[offset + d];
            let right = input[offset + d + 64];
            let at = row * 2048 + head * 128 + d;
            // Keep the original rounded sine product before the fused multiply-add.
            // Explicit instructions prevent contraction from depending on register use.
            unsafe {
                *query.add(at) = cuda_device::float::fma_rn_f32(
                    left,
                    cosine,
                    -cuda_device::float::mul_rn_f32(right, sine),
                );
                *query.add(at + 64) = cuda_device::float::fma_rn_f32(
                    right,
                    cosine,
                    cuda_device::float::mul_rn_f32(left, sine),
                );
            }
            if head < 2 {
                let offset = row * 2560 + 2048 + head * 128;
                let left = input[offset + d];
                let right = input[offset + d + 64];
                let k0 = cuda_device::float::fma_rn_f32(
                    left,
                    cosine,
                    -cuda_device::float::mul_rn_f32(right, sine),
                );
                let k1 = cuda_device::float::fma_rn_f32(
                    right,
                    cosine,
                    cuda_device::float::mul_rn_f32(left, sine),
                );
                let position = positions[row] as usize;
                let page = pages[sequences[row] as usize * stride + position / 32] as usize;
                let at = (page * 32 + position % 32) * 256 + head * 128 + d;
                unsafe {
                    *keys.add(at) = (k0 as f16).to_bits();
                    *keys.add(at + 64) = (k1 as f16).to_bits();
                    *values.add(at) = (input[offset + 256 + d] as f16).to_bits();
                    *values.add(at + 64) = (input[offset + 320 + d] as f16).to_bits();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::gpu::kernels;
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};

    #[test]
    #[ignore = "requires CUDA"]
    fn combined_rotary_cache_matches_separate_operators() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        let positions = [0_u32, 31, 32, 8191, 131071];
        let sequences = [0_u32, 1, 0, 1, 0];
        let mut tables = vec![0_u32; 2 * 4096];
        for (row, &page) in [5, 3, 2, 7, 1].iter().enumerate() {
            tables[sequences[row] as usize * 4096 + positions[row] as usize / 32] = page;
        }
        let q: Vec<_> = (0..5 * 2048)
            .map(|i| (i * 37 % 1021) as f32 / 128.0 - 4.0)
            .collect();
        let k: Vec<_> = (0..5 * 256)
            .map(|i| (i * 43 % 997) as f32 / 256.0 - 2.0)
            .collect();
        let v: Vec<_> = (0..5 * 256)
            .map(|i| (i * 17 % 509) as f32 / 128.0 - 2.0)
            .collect();
        let mut combined = Vec::new();
        for row in 0..5 {
            combined.extend_from_slice(&q[row * 2048..(row + 1) * 2048]);
            combined.extend_from_slice(&k[row * 256..(row + 1) * 256]);
            combined.extend_from_slice(&v[row * 256..(row + 1) * 256]);
        }
        let q = DeviceBuffer::from_host(&stream, &q)?;
        let k = DeviceBuffer::from_host(&stream, &k)?;
        let v = DeviceBuffer::from_host(&stream, &v)?;
        let input = DeviceBuffer::from_host(&stream, &combined)?;
        let positions = DeviceBuffer::from_host(&stream, &positions)?;
        let sequences = DeviceBuffer::from_host(&stream, &sequences)?;
        let tables = DeviceBuffer::from_host(&stream, &tables)?;
        let coefficients = DeviceBuffer::<f32>::zeroed(&stream, 5 * 128)?;
        let qr = DeviceBuffer::<f32>::zeroed(&stream, 5 * 2048)?;
        let kr = DeviceBuffer::<f32>::zeroed(&stream, 5 * 256)?;
        let query = DeviceBuffer::<f32>::zeroed(&stream, 5 * 2048)?;
        let keys = DeviceBuffer::<u16>::zeroed(&stream, 8 * 32 * 256)?;
        let values = DeviceBuffer::<u16>::zeroed(&stream, keys.len())?;
        let expected_k = DeviceBuffer::<u16>::zeroed(&stream, keys.len())?;
        let expected_v = DeviceBuffer::<u16>::zeroed(&stream, keys.len())?;
        // SAFETY: complete matrix rows and unique, bounded physical KV writes;
        // every buffer remains live through the synchronized host comparisons.
        unsafe {
            module.llama.llama_rope(
                &stream,
                LaunchConfig::for_num_elems(q.len() as u32),
                16,
                0,
                5_000_000.0,
                true,
                &positions,
                &q,
                qr.cu_deviceptr() as *mut f32,
            )?;
            module.llama.llama_rope(
                &stream,
                LaunchConfig::for_num_elems(k.len() as u32),
                2,
                0,
                5_000_000.0,
                true,
                &positions,
                &k,
                kr.cu_deviceptr() as *mut f32,
            )?;
            module.llama.llama_cache_paged(
                &stream,
                LaunchConfig::for_num_elems(k.len() as u32),
                4096,
                &positions,
                &sequences,
                &tables,
                &kr,
                &v,
                expected_k.cu_deviceptr() as *mut u16,
                expected_v.cu_deviceptr() as *mut u16,
            )?;
            module.llama_position.coefficients(
                &stream,
                LaunchConfig::for_num_elems(5 * 64),
                5,
                5_000_000.0,
                &positions,
                coefficients.cu_deviceptr() as *mut f32,
            )?;
            module.llama_position.rotary_cache(
                &stream,
                LaunchConfig::for_num_elems(5 * 2048),
                5,
                4096,
                &positions,
                &sequences,
                &tables,
                &coefficients,
                &input,
                query.cu_deviceptr() as *mut f32,
                keys.cu_deviceptr() as *mut u16,
                values.cu_deviceptr() as *mut u16,
            )?;
        }
        let bits = |values: Vec<f32>| values.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        assert_eq!(
            bits(query.to_host_vec(&stream)?),
            bits(qr.to_host_vec(&stream)?)
        );
        assert_eq!(keys.to_host_vec(&stream)?, expected_k.to_host_vec(&stream)?);
        assert_eq!(
            values.to_host_vec(&stream)?,
            expected_v.to_host_vec(&stream)?
        );
        Ok(())
    }
    #[test]
    #[ignore = "requires external official rotary coefficients and CUDA"]
    fn rotary_coefficients_match_independent_long_positions() -> Result<()> {
        use anyhow::{Context, ensure};
        let path = std::env::var("LIA_TEST_MINICPM5_ROPE_REFERENCE")
            .context("set the external official rotary reference path")?;
        let reference: serde_json::Value = serde_json::from_reader(std::fs::File::open(path)?)?;
        let positions: Vec<u32> = serde_json::from_value(reference["positions"].clone())?;
        let expected: Vec<f32> = serde_json::from_value(reference["coefficients"].clone())?;
        ensure!(
            expected.len() == positions.len() * 128 && positions.iter().any(|&p| p >= 131071),
            "incomplete long-position rotary oracle"
        );
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        let positions = DeviceBuffer::from_host(&stream, &positions)?;
        let output = DeviceBuffer::<f32>::zeroed(&stream, expected.len())?;
        // SAFETY: complete coefficient rows, bounded metadata, live buffers.
        unsafe {
            module.llama_position.coefficients(
                &stream,
                LaunchConfig::for_num_elems((positions.len() * 64) as u32),
                positions.len(),
                5_000_000.0,
                &positions,
                output.cu_deviceptr() as *mut f32,
            )?;
        }
        let mut maximum = 0_f32;
        for (actual, expected) in output.to_host_vec(&stream)?.into_iter().zip(expected) {
            ensure!(
                actual.is_finite() && expected.is_finite(),
                "nonfinite coefficient"
            );
            maximum = maximum.max((actual - expected).abs());
        }
        println!("independent rotary coefficients max_abs={maximum}");
        ensure!(
            maximum < 5e-7,
            "rotary coefficients differ from official reference"
        );
        Ok(())
    }
}
