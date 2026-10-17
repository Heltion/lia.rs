use super::*;

/// # Safety
/// Buffers must remain live on the stream and output must cover rows * batch.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch_prefill_matrix(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    kind: u32,
    cols: usize,
    rows: usize,
    batch: usize,
    weights: &DeviceBuffer<u8>,
    input: &DeviceBuffer<f32>,
    workspace: &DeviceBuffer<u32>,
    packed_input: &DeviceBuffer<u32>,
    output: *mut f32,
) -> anyhow::Result<()> {
    let (elements, bytes) =
        crate::gguf::layout(kind).ok_or_else(|| anyhow!("unsupported prefill matrix format"))?;
    anyhow::ensure!(
        rows.checked_mul(cols)
            .is_some_and(|n| n <= u32::MAX as usize)
            && batch
                .checked_mul(rows)
                .is_some_and(|n| n <= u32::MAX as usize)
            && batch
                .checked_mul(cols)
                .is_some_and(|n| n <= u32::MAX as usize)
            && cols > 0
            && rows > 0
            && cols.is_multiple_of(256)
            && (1..=512).contains(&batch)
            && (batch < 64 || workspace.len() >= rows * cols / 2)
            && (packed_input.cu_deviceptr() as usize).is_multiple_of(16)
            && (batch < 64 || (workspace.cu_deviceptr() as usize).is_multiple_of(16))
            && packed_input.len() >= batch * cols / 2
            && input.len() == batch * cols
            && weights.len() == rows * cols / elements as usize * bytes as usize,
        "prefill matrix shape mismatch"
    );
    // SAFETY: the packed allocation covers every complete activation pair.
    unsafe {
        module.tensor.pack_input(
            stream,
            LaunchConfig::for_num_elems((input.len() / 2) as u32),
            input,
            packed_input.cu_deviceptr() as *mut u32,
        )?;
    }
    if batch < 64 {
        let config = LaunchConfig {
            grid_dim: ((rows.div_ceil(64) * batch.div_ceil(32)) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        macro_rules! launch {
            ($kind:literal, $bytes:literal) => {
                module.packed_prefill.prefill_packed::<$kind, $bytes>(
                    stream,
                    config,
                    cols,
                    rows,
                    batch,
                    weights,
                    packed_input,
                    output,
                )?
            };
        }
        // SAFETY: full blocks and complete CTAs; all tails are padded and guarded.
        unsafe {
            match kind {
                0 => launch!(0, 4),
                30 => launch!(30, 2),
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
                _ => return Err(anyhow!("unsupported packed prefill format")),
            }
        }
        return Ok(());
    }
    let config = LaunchConfig {
        grid_dim: ((rows.div_ceil(128) * batch.div_ceil(128)) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    macro_rules! launch {
        ($kind:literal, $bytes:literal) => {
            module.tensor.unpack_matrix::<$kind, $bytes>(
                stream,
                LaunchConfig::for_num_elems((rows * cols / 8) as u32),
                weights,
                workspace.cu_deviceptr() as *mut u32,
                rows * cols,
            )?
        };
    }
    // SAFETY: validated complete blocks, full CTAs, padded tail rows and tokens.
    unsafe {
        match kind {
            0 => launch!(0, 4),
            30 => launch!(30, 2),
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
            _ => return Err(anyhow!("unsupported prefill matrix format")),
        }
    }
    // SAFETY: same validated matrix and output; unpack completes first on this stream.
    unsafe {
        module.tensor.prefill_matrix(
            stream,
            config,
            cols,
            rows,
            batch,
            workspace,
            packed_input,
            output,
        )?;
    }
    Ok(())
}
