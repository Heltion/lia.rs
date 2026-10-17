//! Rust GPU operators and host launch validation.
use anyhow::anyhow;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, simt::LaunchConfig};
mod dispatch;
mod inspection;
pub(crate) mod kernels;
#[cfg(test)]
use dispatch::launch_matvec;
pub(crate) use dispatch::{launch_int_matmul, launch_matmul};
pub use inspection::kernel_info;

mod tensor;
pub(crate) use tensor::launch_prefill_matrix;
pub(crate) use tensor::{DENSE_MIN_BATCH, WIDE_MIN_BATCH};

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    #[test]
    fn scalar_operators() -> anyhow::Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let n = 1003;
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.125 - 63.0).collect();
        let y: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * -0.25).collect();
        let dx = DeviceBuffer::from_host(&stream, &x)?;
        let dy = DeviceBuffer::from_host(&stream, &y)?;
        let mut out = DeviceBuffer::<f32>::zeroed(&stream, n)?;
        let module = kernels::load(&context)?;
        // SAFETY: the one-dimensional grid covers the output; the kernel guards its
        // tail and every input has exactly the output length.
        unsafe {
            module.basic.residual(
                &stream,
                LaunchConfig::for_num_elems(n as u32),
                &dx,
                &dy,
                &mut out,
            )?;
        }
        let actual = out.to_host_vec(&stream)?;
        for (i, actual) in actual.iter().enumerate() {
            if *actual != x[i] + y[i] {
                return Err(anyhow!("GPU residual mismatch at {i}"));
            }
        }
        println!("GPU residual: {n} elements match the Rust CPU reference exactly");
        let formats = [
            (10, 84),
            (12, 144),
            (14, 210),
            (16, 66),
            (17, 74),
            (18, 98),
            (19, 50),
            (21, 110),
            (22, 82),
            (23, 136),
            (29, 56),
        ];
        for (kind, block_bytes) in formats {
            let mut input: Vec<u8> = (0..block_bytes * 3)
                .map(|i| ((i * 73 + 19) % 256) as u8)
                .collect();
            for block in input.chunks_exact_mut(block_bytes) {
                let offset = if kind == 10 {
                    80
                } else if kind == 14 {
                    208
                } else {
                    0
                };
                if kind != 29 {
                    block[offset..offset + 2].copy_from_slice(&0x3800u16.to_le_bytes());
                    if kind == 10 || kind == 12 {
                        block[offset + 2..offset + 4].copy_from_slice(&0x3400u16.to_le_bytes());
                    }
                } else {
                    for (j, nibble) in [0u16, 0, 8, 3].iter().enumerate() {
                        let pos = 48 + j * 2;
                        let bits = u16::from_le_bytes([block[pos], block[pos + 1]]);
                        block[pos..pos + 2]
                            .copy_from_slice(&((bits & 4095) | (nibble << 12)).to_le_bytes());
                    }
                }
            }
            let mut expected = Vec::new();
            for block in input.chunks_exact(block_bytes) {
                expected.extend(crate::quant::decode_block(kind, block)?);
            }
            let din = DeviceBuffer::from_host(&stream, &input)?;
            let mut dout = DeviceBuffer::<f32>::zeroed(&stream, expected.len())?;
            // SAFETY: input has three complete blocks, output has three groups of 256;
            // the one-dimensional grid matches the guarded output index.
            unsafe {
                module.basic.decode(
                    &stream,
                    LaunchConfig::for_num_elems(expected.len() as u32),
                    kind,
                    block_bytes,
                    &din,
                    &mut dout,
                )?;
            }
            let actual = dout.to_host_vec(&stream)?;
            let mut max_error = 0.0f32;
            for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
                let error = (a - e).abs();
                if !a.is_finite() || !e.is_finite() || error > 1e-5 * e.abs().max(1.0) {
                    return Err(anyhow!(
                        "GPU storage type {kind} mismatch at {i}: {a} vs {e}"
                    ));
                }
                max_error = max_error.max(error);
            }
            println!(
                "GPU storage type {kind}: {} values, maximum error {max_error}",
                expected.len()
            );
            let x: Vec<f32> = (0..256).map(|j| (j as f32 - 117.0) * 0.001).collect();
            let dx = DeviceBuffer::from_host(&stream, &x)?;
            let dout = DeviceBuffer::<f32>::zeroed(&stream, 3)?;
            // SAFETY: one complete warp per row, all inputs cover three 256-element
            // rows, and each row is written once by its warp's lane zero.
            unsafe {
                module.matrix.matvec(
                    &stream,
                    LaunchConfig::for_num_elems(96),
                    kind,
                    block_bytes,
                    256,
                    256,
                    3,
                    &din,
                    &dx,
                    dout.cu_deviceptr() as *mut f32,
                )?;
            }
            let actual = dout.to_host_vec(&stream)?;
            for (row, a) in actual.iter().enumerate() {
                let expected: f64 = expected[row * 256..(row + 1) * 256]
                    .iter()
                    .zip(&x)
                    .map(|(w, x)| *w as f64 * *x as f64)
                    .sum();
                if (*a as f64 - expected).abs() > 2e-4 * expected.abs().max(1.0) {
                    return Err(anyhow!(
                        "GPU matvec type {kind}, row {row}: {a} vs {expected}"
                    ));
                }
            }
            println!("GPU matvec type {kind}: three rows match f64 reference");
        }
        let dim = 65;
        let groups = 3;
        let input: Vec<f32> = (0..dim * groups)
            .map(|i| (i as f32 - 101.0) * 0.003)
            .collect();
        let weight: Vec<f32> = (0..dim).map(|i| 0.75 + i as f32 * 0.002).collect();
        let dx = DeviceBuffer::from_host(&stream, &input)?;
        let dw = DeviceBuffer::from_host(&stream, &weight)?;
        let output = DeviceBuffer::<f32>::zeroed(&stream, input.len())?;
        // SAFETY: each full warp owns a group; input and output cover three groups,
        // and every lane only writes its own strided elements.
        unsafe {
            module.normalization.rms_norm(
                &stream,
                LaunchConfig::for_num_elems((groups * 32) as u32),
                dim,
                groups,
                1e-6,
                &dx,
                &dw,
                output.cu_deviceptr() as *mut f32,
            )?;
        }
        let actual = output.to_host_vec(&stream)?;
        for group in 0..groups {
            let sum: f64 = input[group * dim..(group + 1) * dim]
                .iter()
                .map(|x| (*x as f64).powi(2))
                .sum();
            let scale = 1.0 / (sum / dim as f64 + 1e-6).sqrt();
            for j in 0..dim {
                let expected = input[group * dim + j] as f64 * scale * weight[j] as f64;
                if (actual[group * dim + j] as f64 - expected).abs() > 1e-5 {
                    return Err(anyhow!("GPU RMS norm mismatch"));
                }
            }
        }
        println!("GPU RMS norm: three non-aligned groups match f64 reference");
        Ok(())
    }

    pub(crate) fn close(actual: &[f32], expected: &[f64], name: &str) -> Result<()> {
        if actual.len() != expected.len() {
            return Err(anyhow!("{name}: length mismatch"));
        }
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            if !a.is_finite() || (*a as f64 - e).abs() > 2e-5 * e.abs().max(1.0) {
                return Err(anyhow!("{name}[{i}]: {a} != {e}"));
            }
        }
        Ok(())
    }

    /// Validate real model matrix rows against an independent f64 host dot product.
    #[test]
    #[ignore = "requires LIA_TEST_MODEL and a CUDA device"]
    fn model_matrix_rows() -> anyhow::Result<()> {
        let path = std::env::var("LIA_TEST_MODEL")?;
        use std::io::{Read, Seek, SeekFrom};
        let model = crate::gguf::Model::open(&path)?;
        let mut file = std::fs::File::open(&path)?;
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        let mut seen = std::collections::BTreeSet::new();
        for tensor in &model.tensors {
            if tensor.shape.len() != 2 || !seen.insert(tensor.kind) {
                continue;
            }
            let (block_elems, block_bytes) = crate::gguf::layout(tensor.kind)
                .ok_or_else(|| anyhow!("unsupported storage layout"))?;
            let cols = tensor.shape[0] as usize;
            let rows = (tensor.shape[1] as usize).min(3);
            let row_bytes = cols / block_elems as usize * block_bytes as usize;
            let mut input = vec![0u8; rows * row_bytes];
            file.seek(SeekFrom::Start(model.data_offset + tensor.offset))?;
            file.read_exact(&mut input)?;
            let x: Vec<f32> = (0..cols)
                .map(|i| ((i * 31 % 127) as f32 - 63.0) * 0.001)
                .collect();
            let integer =
                std::env::var_os("LIA_INT8").is_some() && tensor.kind != 0 && tensor.kind != 30;
            let expected_x: Vec<f32> = if integer {
                let mut values = Vec::new();
                for group in x.as_chunks::<32>().0 {
                    let max = group.iter().map(|v| v.abs()).fold(0f32, f32::max);
                    let scale = if max == 0.0 { 1.0 } else { max / 127.0 };
                    values.extend(group.iter().map(|v| (v / scale).round_ties_even() * scale));
                }
                values
            } else {
                x.clone()
            };
            let mut expected = Vec::new();
            for row in input.chunks_exact(row_bytes) {
                let mut sum = 0.0f64;
                let mut j = 0;
                for block in row.chunks_exact(block_bytes as usize) {
                    for value in crate::quant::decode_block(tensor.kind, block)? {
                        sum += value as f64 * expected_x[j] as f64;
                        j += 1;
                    }
                }
                expected.push(sum);
            }
            let weights = DeviceBuffer::from_host(&stream, &input)?;
            let dx = DeviceBuffer::from_host(&stream, &x)?;
            let out = DeviceBuffer::<f32>::zeroed(&stream, rows)?;
            // SAFETY: dimensions and byte extents came from the validated container;
            // rows have full storage blocks, and each warp writes its own row.
            if integer {
                let packed = DeviceBuffer::<u32>::zeroed(&stream, cols / 4)?;
                let scales = DeviceBuffer::<f32>::zeroed(&stream, cols / 32)?;
                // SAFETY: the quantized matrix has complete activation groups and
                // correctly sized persistent inputs/outputs through this copy.
                unsafe {
                    module.integer.quantize_activation(
                        &stream,
                        LaunchConfig::for_num_elems(cols as u32),
                        &dx,
                        packed.cu_deviceptr() as *mut u32,
                        scales.cu_deviceptr() as *mut f32,
                    )?;
                    launch_int_matmul(
                        &module,
                        &stream,
                        tensor.kind,
                        cols,
                        rows,
                        1,
                        1,
                        1,
                        128,
                        &weights,
                        &packed,
                        &scales,
                        out.cu_deviceptr() as *mut f32,
                    )?;
                }
                stream.synchronize()?;
            } else {
                unsafe {
                    launch_matvec(
                        &module,
                        &stream,
                        tensor.kind,
                        cols,
                        rows,
                        &weights,
                        &dx,
                        out.cu_deviceptr() as *mut f32,
                    )?;
                }
            }
            let actual = out.to_host_vec(&stream)?;
            let mut max_error = 0.0f64;
            for (a, e) in actual.iter().zip(&expected) {
                let error = (*a as f64 - *e).abs();
                if !a.is_finite() || error > 1e-4 * e.abs().max(1.0) {
                    return Err(anyhow!("model matrix {} mismatch: {a} vs {e}", tensor.name));
                }
                max_error = max_error.max(error);
            }
            println!(
                "{} type={} shape={:?} checked_rows={rows} integer={integer} max_error={max_error}",
                tensor.name, tensor.kind, tensor.shape
            );
        }
        Ok(())
    }
}
