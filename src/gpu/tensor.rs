use super::*;

pub(crate) const MEDIUM_MIN_BATCH: usize = 33;
pub(crate) const WIDE_MIN_BATCH: usize = 65;

/// # Safety
/// Buffers must remain live on the stream and output must cover rows * batch.
/// When reusing, packed_input must already contain this exact immutable input.
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
    reuse_input: bool,
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
            && (packed_input.cu_deviceptr() as usize).is_multiple_of(16)
            && packed_input.len() >= batch * cols / 2
            && input.len() == batch * cols
            && weights.len() == rows * cols / elements as usize * bytes as usize,
        "prefill matrix shape mismatch"
    );
    // SAFETY: the packed allocation covers every complete activation pair.
    if !reuse_input {
        unsafe {
            module.tensor.pack_input(
                stream,
                LaunchConfig::for_num_elems((input.len() / 2) as u32),
                input,
                packed_input.cu_deviceptr() as *mut u32,
            )?;
        }
    }
    let tokens = if batch >= WIDE_MIN_BATCH {
        128
    } else if batch >= MEDIUM_MIN_BATCH {
        64
    } else {
        32
    };
    let config = LaunchConfig {
        grid_dim: ((rows.div_ceil(64) * batch.div_ceil(tokens)) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    macro_rules! launch {
        ($kind:literal, $bytes:literal) => {
            if batch >= WIDE_MIN_BATCH {
                module.wide_prefill.prefill_wide::<$kind, $bytes, 128>(
                    stream,
                    config,
                    cols,
                    rows,
                    batch,
                    weights,
                    packed_input,
                    output,
                )?;
            } else if batch >= MEDIUM_MIN_BATCH {
                module.wide_prefill.prefill_wide::<$kind, $bytes, 64>(
                    stream,
                    config,
                    cols,
                    rows,
                    batch,
                    weights,
                    packed_input,
                    output,
                )?;
            } else {
                module.packed_prefill.prefill_packed::<$kind, $bytes>(
                    stream,
                    config,
                    cols,
                    rows,
                    batch,
                    weights,
                    packed_input,
                    output,
                )?;
            }
        };
    }
    // SAFETY: complete CTAs, full quantization blocks, guarded row/token tails.
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::gpu::kernels;
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer};

    #[test]
    fn prefill_matrix() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        let (cols, rows) = (768, 143);
        for batch in [
            9, 31, 32, 33, 63, 64, 65, 66, 96, 97, 127, 128, 129, 255, 256, 257, 512,
        ] {
            let input: Vec<f32> = (0..cols * batch)
                .map(|i| ((i * 71 % 257) as f32 - 128.0) * 0.003)
                .collect();
            let packed_input = DeviceBuffer::<u32>::zeroed(&stream, input.len() / 2)?;
            let gpu_input = DeviceBuffer::from_host(&stream, &input)?;
            let mut reuse_input = false;
            for (kind, size) in [
                (0, 4),
                (30, 2),
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
                let block_size = if kind == 0 || kind == 30 { 1 } else { 256 };
                let mut weights: Vec<u8> = (0..rows * (cols / block_size) * size)
                    .map(|i| ((i * 73 + 19) % 256) as u8)
                    .collect();
                if kind == 0 || kind == 30 {
                    weights.clear();
                    for i in 0..rows * cols {
                        let value = ((i * 17 % 37) as f32 - 18.0) * 0.03;
                        if kind == 0 {
                            weights.extend_from_slice(&value.to_le_bytes());
                        } else {
                            weights
                                .extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
                        }
                    }
                } else {
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
                                b[p..p + 2]
                                    .copy_from_slice(&((v & 4095) | (nib << 12)).to_le_bytes());
                            }
                        } else {
                            b[offset..offset + 2].copy_from_slice(&0x3800u16.to_le_bytes());
                            if kind == 10 || kind == 12 {
                                b[offset + 2..offset + 4].copy_from_slice(&0x3400u16.to_le_bytes());
                            }
                        }
                    }
                }

                let gpu_weights = DeviceBuffer::from_host(&stream, &weights)?;
                let output = DeviceBuffer::<f32>::zeroed(&stream, rows * batch)?;
                // SAFETY: complete matrices, intentionally partial row/token tiles,
                // and output covering every final element.
                unsafe {
                    crate::gpu::launch_prefill_matrix(
                        &module,
                        &stream,
                        kind,
                        cols,
                        rows,
                        batch,
                        &gpu_weights,
                        &gpu_input,
                        reuse_input,
                        &packed_input,
                        output.cu_deviceptr() as *mut f32,
                    )?;
                }
                reuse_input = true;
                let actual = output.to_host_vec(&stream)?;
                for row in 0..rows {
                    let mut decoded = Vec::new();
                    for block in weights
                        [row * cols / block_size * size..(row + 1) * cols / block_size * size]
                        .chunks_exact(size)
                    {
                        decoded.extend(crate::quant::decode_block(kind, block)?);
                    }
                    for token in 0..batch {
                        let mut expected = 0.0f64;
                        let mut magnitude = 0.0f64;
                        for j in 0..cols {
                            let product = (decoded[j] as f16 as f32) as f64
                                * (input[token * cols + j] as f16 as f32) as f64;
                            expected += product;
                            magnitude += product.abs();
                        }
                        let value = actual[token * rows + row];
                        anyhow::ensure!(
                            value.is_finite()
                                && (value as f64 - expected).abs() <= 2e-5 * magnitude.max(1.0),
                            "prefill matrix kind={kind} batch={batch} row={row} token={token}: {value} vs {expected}"
                        );
                    }
                }
                println!(
                    "GPU prefill matrix kind={kind} batch={batch} matches f64 reference with f16 inputs and padded tiles"
                );
            }
        }
        Ok(())
    }
}
