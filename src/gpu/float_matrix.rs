//! Checked launch geometry for the codec's Float32 tensor products.
use super::*;
use anyhow::{Result, ensure};

pub(crate) fn launch(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    cols: usize,
    rows: usize,
    batch: usize,
    weights: &DeviceBuffer<u8>,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
) -> Result<()> {
    ensure!(
        cols > 0
            && cols.is_multiple_of(32)
            && rows > 0
            && (1..=crate::MAX_CONTEXT_TOKENS).contains(&batch)
            && rows.checked_mul(cols).and_then(|n| n.checked_mul(4)) == Some(weights.len())
            && batch.checked_mul(cols) == Some(input.len())
            && batch.checked_mul(rows) == Some(output.len())
            && output.len() <= u32::MAX as usize,
        "Float32 matrix shape mismatch"
    );
    // SAFETY: complete 32-column tiles, four complete warps per block, zero-padded
    // row/token tails and a separately allocated output with checked extents.
    unsafe {
        module.float_matrix.float_matrix(
            stream,
            LaunchConfig {
                grid_dim: ((rows.div_ceil(32) * batch.div_ceil(32)) as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            },
            cols,
            rows,
            batch,
            weights,
            input,
            output.cu_deviceptr() as *mut f32,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn check(weights: &[f32], rows: usize, cols: usize, batch: usize) -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        let input: Vec<f32> = (0..batch * cols)
            .map(|i| ((i * 73 % 257) as f32 - 128.) * 0.0137)
            .collect();
        let bytes: Vec<u8> = weights.iter().flat_map(|v| v.to_le_bytes()).collect();
        let w = DeviceBuffer::from_host(&stream, &bytes)?;
        let x = DeviceBuffer::from_host(&stream, &input)?;
        let mut y = DeviceBuffer::<f32>::zeroed(&stream, rows * batch)?;
        launch(&module, &stream, cols, rows, batch, &w, &x, &mut y)?;
        let actual = y.to_host_vec(&stream)?;
        let mut maximum = 0.0f64;
        for token in 0..batch {
            for row in 0..rows {
                let expected: f64 = (0..cols)
                    .map(|k| {
                        f64::from(weights[row * cols + k]) * f64::from(input[token * cols + k])
                    })
                    .sum();
                let error = (f64::from(actual[token * rows + row]) - expected).abs();
                maximum = maximum.max(error);
                ensure!(
                    error < 0.00003 * (1.0 + expected.abs()),
                    "Float32 product mismatch token={token} row={row}: {} != {expected}",
                    actual[token * rows + row]
                );
            }
        }
        println!("Float32 matrix rows={rows} cols={cols} batch={batch}: max_abs={maximum}");
        Ok(())
    }
    #[test]
    fn tiled_products_match_f64_with_partial_rows_and_tokens() -> Result<()> {
        for (rows, cols, batch) in [(1, 32, 1), (17, 64, 3), (35, 256, 9), (67, 512, 37)] {
            let weights: Vec<f32> = (0..rows * cols)
                .map(|i| ((i * 31 % 251) as f32 - 125.) * 0.00371)
                .collect();
            check(&weights, rows, cols, batch)?;
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires official flow.pt and CUDA"]
    fn real_codec_matrix_matches_f64() -> Result<()> {
        let path = std::env::var("LIA_TOKEN2WAV_FLOW")?;
        let archive = crate::torch_weights::Archive::open(path)?;
        let name = archive
            .tensors()
            .iter()
            .find(|(_, t)| t.shape == [512, 512])
            .map(|(name, _)| name)
            .ok_or_else(|| anyhow!("missing codec projection"))?;
        let bytes = archive.read(name)?;
        let weights: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        println!("official codec matrix: {name}");
        check(&weights, 512, 512, 37)
    }
}
