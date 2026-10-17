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
        || x.len() != batch * cols / 4
        || (row_group != 1 && row_group != 2)
        || (split == 4 && row_group != 2)
        || (split != 1 && split != 4)
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
        LaunchConfig::for_num_elems((rows.div_ceil(row_group) * 32) as u32)
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
