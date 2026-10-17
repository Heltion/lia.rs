use super::*;

/// Validate real model matrix rows against an independent f64 host dot product.
pub fn check_model(path: &str) -> anyhow::Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    let model = crate::gguf::Model::open(path)?;
    let mut file = std::fs::File::open(path)?;
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
