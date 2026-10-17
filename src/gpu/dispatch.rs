use super::*;

/// Select a compile-time storage specialization after checking host dimensions.
///
/// # Safety
/// Output must cover `rows` floats and remain alive until this stream completes.
/// Input must cover `cols` floats; weights must contain `rows * cols` elements.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch_matvec(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    kind: u32,
    cols: usize,
    rows: usize,
    weights: &DeviceBuffer<u8>,
    x: &DeviceBuffer<f32>,
    out: *mut f32,
) -> anyhow::Result<()> {
    let config = LaunchConfig::for_num_elems((rows * 32) as u32);
    let (elems, bytes) =
        crate::gguf::layout(kind).ok_or_else(|| anyhow!("invalid matrix storage type"))?;
    if x.len() != cols
        || !cols.is_multiple_of(elems as usize)
        || weights.len() != rows * (cols / elems as usize) * bytes as usize
    {
        return Err(anyhow!("matvec launch shape mismatch"));
    }
    // SAFETY: caller guarantees output lifetime and extent; checked input shapes
    // and a full-warp one-dimensional launch satisfy each specialization.
    unsafe {
        match kind {
            0 => module
                .matrix
                .matvec_packed::<0, 4, 1>(stream, config, cols, rows, weights, x, out)?,
            30 => module
                .matrix
                .matvec_packed::<30, 2, 1>(stream, config, cols, rows, weights, x, out)?,
            10 => module
                .matrix
                .matvec_packed::<10, 84, 256>(stream, config, cols, rows, weights, x, out)?,
            12 => module
                .matrix
                .matvec_packed::<12, 144, 256>(stream, config, cols, rows, weights, x, out)?,
            14 => module
                .matrix
                .matvec_packed::<14, 210, 256>(stream, config, cols, rows, weights, x, out)?,
            16 => module
                .matrix
                .matvec_packed::<16, 66, 256>(stream, config, cols, rows, weights, x, out)?,
            17 => module
                .matrix
                .matvec_packed::<17, 74, 256>(stream, config, cols, rows, weights, x, out)?,
            18 => module
                .matrix
                .matvec_packed::<18, 98, 256>(stream, config, cols, rows, weights, x, out)?,
            19 => module
                .matrix
                .matvec_packed::<19, 50, 256>(stream, config, cols, rows, weights, x, out)?,
            21 => module
                .matrix
                .matvec_packed::<21, 110, 256>(stream, config, cols, rows, weights, x, out)?,
            22 => module
                .matrix
                .matvec_packed::<22, 82, 256>(stream, config, cols, rows, weights, x, out)?,
            23 => module
                .matrix
                .matvec_packed::<23, 136, 256>(stream, config, cols, rows, weights, x, out)?,
            29 => module
                .matrix
                .matvec_packed::<29, 56, 256>(stream, config, cols, rows, weights, x, out)?,
            _ => return Err(anyhow!("unsupported matvec storage type")),
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch_matmul(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    kind: u32,
    cols: usize,
    rows: usize,
    batch: usize,
    weights: &DeviceBuffer<u8>,
    x: &DeviceBuffer<f32>,
    out: *mut f32,
) -> anyhow::Result<()> {
    if batch == 1 {
        return unsafe { launch_matvec(module, stream, kind, cols, rows, weights, x, out) };
    }
    let (elems, bytes) =
        crate::gguf::layout(kind).ok_or_else(|| anyhow!("invalid matrix format"))?;
    if batch == 0
        || batch > 8
        || x.len() != batch * cols
        || !cols.is_multiple_of(elems as usize)
        || weights.len() != rows * (cols / elems as usize) * bytes as usize
    {
        return Err(anyhow!("batched matrix launch shape mismatch"));
    }
    let config = LaunchConfig::for_num_elems((rows * 32) as u32);
    // SAFETY: caller owns the output extent and lifetime. Input shapes are
    // checked above; each warp owns one row in every token's output matrix.
    unsafe {
        match kind {
            0 => module
                .matrix
                .matmul_eight::<0, 4, 1>(stream, config, cols, rows, batch, weights, x, out)?,
            30 => module
                .matrix
                .matmul_eight::<30, 2, 1>(stream, config, cols, rows, batch, weights, x, out)?,
            10 => module
                .matrix
                .matmul_eight::<10, 84, 256>(stream, config, cols, rows, batch, weights, x, out)?,
            12 => module
                .matrix
                .matmul_eight::<12, 144, 256>(stream, config, cols, rows, batch, weights, x, out)?,
            14 => module
                .matrix
                .matmul_eight::<14, 210, 256>(stream, config, cols, rows, batch, weights, x, out)?,
            16 => module
                .matrix
                .matmul_eight::<16, 66, 256>(stream, config, cols, rows, batch, weights, x, out)?,
            17 => module
                .matrix
                .matmul_eight::<17, 74, 256>(stream, config, cols, rows, batch, weights, x, out)?,
            18 => module
                .matrix
                .matmul_eight::<18, 98, 256>(stream, config, cols, rows, batch, weights, x, out)?,
            19 => module
                .matrix
                .matmul_eight::<19, 50, 256>(stream, config, cols, rows, batch, weights, x, out)?,
            21 => module
                .matrix
                .matmul_eight::<21, 110, 256>(stream, config, cols, rows, batch, weights, x, out)?,
            22 => module
                .matrix
                .matmul_eight::<22, 82, 256>(stream, config, cols, rows, batch, weights, x, out)?,
            23 => module
                .matrix
                .matmul_eight::<23, 136, 256>(stream, config, cols, rows, batch, weights, x, out)?,
            29 => module
                .matrix
                .matmul_eight::<29, 56, 256>(stream, config, cols, rows, batch, weights, x, out)?,
            _ => return Err(anyhow!("unsupported batched matrix type")),
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch_int_matmul(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    kind: u32,
    cols: usize,
    rows: usize,
    batch: usize,
    split: usize,
    row_group: usize,
    block_threads: usize,
    weights: &DeviceBuffer<u8>,
    x: &DeviceBuffer<u32>,
    scales: &DeviceBuffer<f32>,
    out: *mut f32,
) -> anyhow::Result<()> {
    if rows.checked_mul(cols).is_none_or(|n| n > u32::MAX as usize)
        || weights.len() > u32::MAX as usize
    {
        return Err(anyhow!("integer kernel index range exceeded"));
    }
    let (elems, bytes) =
        crate::gguf::layout(kind).ok_or_else(|| anyhow!("invalid integer matrix format"))?;
    if batch == 0
        || batch > 8
        || elems != 256
        || !cols.is_multiple_of(256)
        || !(x.cu_deviceptr() as usize).is_multiple_of(16)
        || x.len() != batch * cols / 4
        || (row_group != 1 && row_group != 2)
        || (split == 4 && row_group != 2)
        || (split != 1 && split != 4)
        || ![64, 128, 256].contains(&block_threads)
        || scales.len() != batch * cols / 32
        || !(weights.cu_deviceptr() as usize).is_multiple_of(256)
        || weights.len() != rows * (cols / 256) * bytes as usize
    {
        return Err(anyhow!("integer matrix shape mismatch"));
    }
    let config = if split == 4 {
        LaunchConfig {
            grid_dim: (rows.div_ceil(row_group) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        }
    } else {
        LaunchConfig {
            grid_dim: (
                (rows.div_ceil(row_group) * 32).div_ceil(block_threads) as u32,
                1,
                1,
            ),
            block_dim: (block_threads as u32, 1, 1),
            shared_mem_bytes: 0,
        }
    };
    // SAFETY: input shapes and full-warp layout were checked above; the caller
    // guarantees the output extent and stream lifetime of all buffers.
    macro_rules! capped {
        ($kind:literal, $bytes:literal, $cap:literal) => {{
            if row_group == 1 {
                module.integer.matmul_int::<$kind, $bytes, $cap, 1, 1>(
                    stream, config, cols, rows, batch, weights, x, scales, out,
                )?;
            } else if split == 4 {
                module.integer.matmul_int::<$kind, $bytes, $cap, 4, 2>(
                    stream, config, cols, rows, batch, weights, x, scales, out,
                )?;
            } else {
                module.integer.matmul_int::<$kind, $bytes, $cap, 1, 2>(
                    stream, config, cols, rows, batch, weights, x, scales, out,
                )?;
            }
        }};
    }
    macro_rules! launch {
        ($kind:literal,$bytes:literal) => {{
            if batch == 1 {
                capped!($kind, $bytes, 1);
            } else if batch <= 4 {
                capped!($kind, $bytes, 4);
            } else {
                capped!($kind, $bytes, 8);
            }
        }};
    }
    // SAFETY: validated matrix extents and one disjoint output row per warp;
    // MAX is a compile-time capacity covering the actual token batch.
    unsafe {
        match kind {
            10 => launch!(10, 84),
            12 => launch!(12, 144),
            14 => launch!(14, 210),
            16 => launch!(16, 66),
            17 => launch!(17, 74),
            18 => launch!(18, 98),
            19 => launch!(19, 50),
            21 => launch!(21, 110),
            22 => launch!(22, 82),
            23 => launch!(23, 136),
            29 => launch!(29, 56),
            _ => return Err(anyhow!("unsupported integer matrix type")),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::gpu::kernels;
    use anyhow::{Result, anyhow};
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};

    #[test]
    fn integer_matmul() -> Result<()> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let module = kernels::load(&ctx)?;
        let cols = 2304;
        let rows = 19;
        for batch in [1, 2, 4, 8] {
            let x: Vec<f32> = (0..batch * cols)
                .map(|i| {
                    if i < 32 {
                        0.0
                    } else {
                        ((i * 71 % 257) as f32 - 128.0) * 0.003
                    }
                })
                .collect();
            let mut quant = Vec::new();
            let mut scales = Vec::new();
            for group in x.as_chunks::<32>().0 {
                let max = group.iter().map(|v| v.abs()).fold(0f32, f32::max);
                let scale = if max == 0.0 { 1.0 } else { max / 127.0 };
                scales.push(scale);
                quant.extend(group.iter().map(|v| (v / scale).round_ties_even() as i8));
            }
            let dx = DeviceBuffer::from_host(&stream, &x)?;
            let packed = DeviceBuffer::<u32>::zeroed(&stream, x.len() / 4)?;
            let ds = DeviceBuffer::<f32>::zeroed(&stream, x.len() / 32)?;
            // SAFETY: complete 32-element groups, four packed bytes per word and one
            // scaling value per group. Every output has a unique producer.
            unsafe {
                module.integer.quantize_activation(
                    &stream,
                    LaunchConfig::for_num_elems(x.len() as u32),
                    &dx,
                    packed.cu_deviceptr() as *mut u32,
                    ds.cu_deviceptr() as *mut f32,
                )?;
            }
            let actual = packed.to_host_vec(&stream)?;
            let expected: Vec<u32> = quant
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| u32::from_le_bytes([v[0] as u8, v[1] as u8, v[2] as u8, v[3] as u8]))
                .collect();
            if actual != expected || ds.to_host_vec(&stream)? != scales {
                return Err(anyhow!("GPU activation quantization mismatch"));
            }
            println!(
                "GPU activation quantization matches nearest-even CPU reference, including all-zero groups"
            );
            for (kind, size) in [
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
            ] {
                let mut weights: Vec<u8> = (0..rows * (cols / 256) * size)
                    .map(|i| ((i * 73 + 19) % 256) as u8)
                    .collect();
                for b in weights.chunks_exact_mut(size) {
                    let offset = if kind == 10 {
                        80
                    } else if kind == 14 {
                        208
                    } else {
                        0
                    };
                    if kind == 29 {
                        for (j, nib) in [0u16, 0, 8, 3].iter().enumerate() {
                            let p = 48 + j * 2;
                            let v = u16::from_le_bytes([b[p], b[p + 1]]);
                            b[p..p + 2].copy_from_slice(&((v & 4095) | (nib << 12)).to_le_bytes());
                        }
                    } else {
                        b[offset..offset + 2].copy_from_slice(&0x3800u16.to_le_bytes());
                        if kind == 10 || kind == 12 {
                            b[offset + 2..offset + 4].copy_from_slice(&0x3400u16.to_le_bytes());
                        }
                    }
                }
                let mut expected = vec![0f64; batch * rows];
                let mut sum_abs = vec![0f64; batch * rows];
                for row in 0..rows {
                    let mut decoded = Vec::new();
                    for block in weights[row * (cols / 256) * size..(row + 1) * (cols / 256) * size]
                        .chunks_exact(size)
                    {
                        decoded.extend(crate::quant::decode_block(kind, block)?);
                    }
                    for t in 0..batch {
                        sum_abs[t * rows + row] = (0..cols)
                            .map(|j| {
                                (decoded[j] as f64
                                    * quant[t * cols + j] as f64
                                    * scales[(t * cols + j) / 32] as f64)
                                    .abs()
                            })
                            .sum();
                        expected[t * rows + row] = (0..cols)
                            .map(|j| {
                                decoded[j] as f64
                                    * quant[t * cols + j] as f64
                                    * scales[(t * cols + j) / 32] as f64
                            })
                            .sum();
                    }
                }
                let dw = DeviceBuffer::from_host(&stream, &weights)?;
                let out = DeviceBuffer::<f32>::zeroed(&stream, batch * rows)?;
                for (threads, split, row_group) in [
                    (64, 1, 1),
                    (128, 1, 1),
                    (256, 1, 1),
                    (64, 1, 2),
                    (128, 1, 2),
                    (256, 1, 2),
                    (128, 4, 2),
                ] {
                    // SAFETY: complete matrices and row-aligned activation groups;
                    // the launch uses one or four complete warps per output row.
                    unsafe {
                        crate::gpu::launch_int_matmul(
                            &module,
                            &stream,
                            kind,
                            cols,
                            rows,
                            batch,
                            split,
                            row_group,
                            threads,
                            &dw,
                            &packed,
                            &ds,
                            out.cu_deviceptr() as *mut f32,
                        )?;
                    }
                    let actual = out.to_host_vec(&stream)?;
                    // Integer dot products are exact. The remaining error is
                    // bounded by rounded scaling, per-lane accumulation and
                    // the warp/CTA reduction tree, including cancellation.
                    let operations = cols.div_ceil(512) + 16;
                    let nu = operations as f64 * f32::EPSILON as f64;
                    let gamma = nu / (1.0 - nu);
                    for (i, ((value, reference), magnitude)) in
                        actual.iter().zip(&expected).zip(&sum_abs).enumerate()
                    {
                        if !value.is_finite()
                            || (*value as f64 - reference).abs() > gamma * magnitude + 1e-6
                        {
                            return Err(anyhow!(
                                "integer matrix rounding bound failed at {i}: {value} vs {reference}"
                            ));
                        }
                    }
                    println!(
                        "GPU integer matrix type {kind}, split {split}, rows {row_group}: {batch} vectors satisfy the f32 bound against the f64 reference"
                    );
                }
            }
        }
        Ok(())
    }
}
