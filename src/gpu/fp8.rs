//! Validated host launches for dynamically normalized FP8 prefill.
use super::*;

pub(crate) fn prepare(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<f32>,
    maxima: &mut DeviceBuffer<f32>,
    scales: &mut DeviceBuffer<f32>,
    packed: &mut DeviceBuffer<u32>,
) -> anyhow::Result<()> {
    let count = input.len().div_ceil(1024);
    anyhow::ensure!(
        count > 0
            && maxima.len() >= count
            && scales.len() >= 2
            && packed.len() >= input.len().div_ceil(4),
        "FP8 preparation workspace mismatch"
    );
    // SAFETY: complete256-thread CTAs, bounded partial maxima and packed extents.
    unsafe {
        module.fp8_prepare.fp8_input_maxima(
            stream,
            LaunchConfig {
                grid_dim: (count as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            input,
            maxima.cu_deviceptr() as *mut f32,
        )?;
        module.fp8_prepare.fp8_input_normalizer(
            stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            count,
            maxima,
            scales.cu_deviceptr() as *mut f32,
        )?;
        module.fp8_prepare.fp8_pack_scaled(
            stream,
            LaunchConfig::for_num_elems(input.len().div_ceil(4) as u32),
            input,
            scales,
            packed,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    kind: u32,
    cols: usize,
    rows: usize,
    batch: usize,
    normalizer: f32,
    weights: &DeviceBuffer<u8>,
    input: &DeviceBuffer<u32>,
    scales: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    cached: Option<&DeviceBuffer<u32>>,
) -> anyhow::Result<()> {
    let (block, bytes) =
        crate::gguf::layout(kind).ok_or_else(|| anyhow!("unsupported FP8 format"))?;
    anyhow::ensure!(
        cols > 0
            && cols.is_multiple_of(256)
            && rows > 0
            && batch > 0
            && normalizer.is_finite()
            && normalizer > 0.0
            && rows
                .checked_mul(cols)
                .is_some_and(|n| n <= u32::MAX as usize)
            && batch
                .checked_mul(cols)
                .is_some_and(|n| n <= u32::MAX as usize)
            && batch
                .checked_mul(rows)
                .is_some_and(|n| n <= u32::MAX as usize)
            && weights.len() == rows * cols / block as usize * bytes as usize
            && input.len() >= batch * cols / 4
            && scales.len() >= 2
            && output.len() == rows * batch,
        "FP8 matrix shape mismatch"
    );
    anyhow::ensure!(
        cached.is_none_or(|b| b.len() >= rows.div_ceil(64) * 64 * cols / 4),
        "FP8 cached weight extent mismatch"
    );
    let config = LaunchConfig {
        grid_dim: ((rows.div_ceil(64) * batch.div_ceil(128)) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    macro_rules! matrix {
        ($kind:literal,$bytes:literal) => {
            module.fp8_matmul.prefill_fp8::<$kind, $bytes, false>(
                stream,
                config,
                cols,
                rows,
                batch,
                normalizer,
                scales,
                weights,
                input,
                input,
                output.cu_deviceptr() as *mut f32,
            )?
        };
    }
    // SAFETY: bounded complete GGUF blocks; tiles zero-pad row/token tails.
    unsafe {
        if let Some(cached) = cached {
            module.fp8_matmul.prefill_fp8::<0, 0, true>(
                stream,
                config,
                cols,
                rows,
                batch,
                normalizer,
                scales,
                weights,
                input,
                cached,
                output.cu_deviceptr() as *mut f32,
            )?;
            return Ok(());
        }
        match kind {
            10 => matrix!(10, 84),
            12 => matrix!(12, 144),
            14 => matrix!(14, 210),
            16 => matrix!(16, 66),
            17 => matrix!(17, 74),
            18 => matrix!(18, 98),
            19 => matrix!(19, 50),
            21 => matrix!(21, 110),
            22 => matrix!(22, 82),
            23 => matrix!(23, 136),
            29 => matrix!(29, 56),
            _ => return Err(anyhow!("unsupported FP8 source format {kind}")),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::kernels::fp8::reference::{decode, encode};
    use anyhow::Result;

    #[test]
    fn dynamic_normalization_handles_tails_and_nonfinite_inputs() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        for n in [1usize, 31, 1024, 1025, 16387] {
            for maximum in [0.0, 0.1, 400.0, 1e30, 1e-30] {
                let values: Vec<f32> = (0..n)
                    .map(|i| if i % 2 == 0 { maximum } else { -maximum * 0.37 })
                    .collect();
                let input = DeviceBuffer::from_host(&stream, &values)?;
                let mut maxima = DeviceBuffer::zeroed(&stream, n.div_ceil(1024))?;
                let mut scales = DeviceBuffer::zeroed(&stream, 2)?;
                let mut packed = DeviceBuffer::zeroed(&stream, n.div_ceil(4))?;
                prepare(
                    &module,
                    &stream,
                    &input,
                    &mut maxima,
                    &mut scales,
                    &mut packed,
                )?;
                let scale = scales.to_host_vec(&stream)?;
                assert_eq!(scale[0].to_bits() & 0x7fffff, 0);
                assert_eq!(scale[0] * scale[1], 1.0);
                let bytes: Vec<u8> = packed
                    .to_host_vec(&stream)?
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                for (i, value) in values.iter().enumerate() {
                    assert!((value * scale[0]).abs() <= 448.0);
                    assert_eq!(bytes[i], encode(value * scale[0]));
                }
                assert!(bytes[n..].iter().all(|&v| v == 0));
            }
        }
        let input = DeviceBuffer::from_host(&stream, &[f32::NAN, 1.0, f32::INFINITY, 0.0])?;
        let mut maxima = DeviceBuffer::zeroed(&stream, 1)?;
        let mut scales = DeviceBuffer::zeroed(&stream, 2)?;
        let mut packed = DeviceBuffer::zeroed(&stream, 1)?;
        prepare(
            &module,
            &stream,
            &input,
            &mut maxima,
            &mut scales,
            &mut packed,
        )?;
        assert!(scales.to_host_vec(&stream)?.iter().all(|v| v.is_nan()));
        Ok(())
    }

    #[test]
    fn tiled_quantized_fp8_matches_independent_f64() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        for (cols, rows, batch) in [(768, 79, 65), (768, 79, 129), (5120, 3, 65), (17408, 3, 65)] {
            let values: Vec<f32> = (0..cols * batch)
                .map(|i| ((i * 71 % 257) as f32 - 128.0) * 0.013)
                .collect();
            let input = DeviceBuffer::from_host(&stream, &values)?;
            let mut maxima = DeviceBuffer::zeroed(&stream, values.len().div_ceil(1024))?;
            let mut scales = DeviceBuffer::zeroed(&stream, 2)?;
            let mut packed = DeviceBuffer::zeroed(&stream, values.len() / 4)?;
            prepare(
                &module,
                &stream,
                &input,
                &mut maxima,
                &mut scales,
                &mut packed,
            )?;
            let scale = scales.to_host_vec(&stream)?;
            let x: Vec<f64> = values
                .iter()
                .map(|v| decode(encode(v * scale[0])) / scale[0] as f64)
                .collect();
            for kind in [10, 12, 14, 16, 17, 18, 19, 21, 22, 23, 29] {
                if cols > 768 && kind != 17 {
                    continue;
                }
                let (_, size) = crate::gguf::layout(kind).unwrap();
                let size = size as usize;
                let mut bytes: Vec<u8> = (0..rows * cols / 256 * size)
                    .map(|i| ((i * 73 + 19) % 256) as u8)
                    .collect();
                for block in bytes.chunks_exact_mut(size) {
                    if kind == 29 {
                        for (j, nibble) in [0u16, 0, 8, 3].iter().enumerate() {
                            let at = 48 + j * 2;
                            let value = u16::from_le_bytes([block[at], block[at + 1]]);
                            block[at..at + 2]
                                .copy_from_slice(&((value & 4095) | (nibble << 12)).to_le_bytes());
                        }
                    } else {
                        let at = if kind == 10 {
                            80
                        } else if kind == 14 {
                            208
                        } else {
                            0
                        };
                        block[at..at + 2].copy_from_slice(&0x3800u16.to_le_bytes());
                        if kind == 10 || kind == 12 {
                            block[at + 2..at + 4].copy_from_slice(&0x3400u16.to_le_bytes());
                        }
                    }
                }
                let normalizer = crate::quant::fp8::normalizer(kind, &bytes)?;
                let mut original = Vec::new();
                for block in bytes.chunks_exact(size) {
                    original.extend(crate::quant::decode_block(kind, block)?);
                }
                let w: Vec<f64> = original
                    .iter()
                    .map(|v| decode(encode(v * normalizer)) / normalizer as f64)
                    .collect();
                let weights = DeviceBuffer::from_host(&stream, &bytes)?;
                let mut output = DeviceBuffer::zeroed(&stream, rows * batch)?;
                launch(
                    &module,
                    &stream,
                    kind,
                    cols,
                    rows,
                    batch,
                    normalizer,
                    &weights,
                    &packed,
                    &scales,
                    &mut output,
                    None,
                )?;
                let actual = output.to_host_vec(&stream)?;
                let cached = super::fp8_cache::pack(
                    &module, &stream, kind, cols, rows, normalizer, &weights,
                )?;
                launch(
                    &module,
                    &stream,
                    kind,
                    cols,
                    rows,
                    batch,
                    normalizer,
                    &weights,
                    &packed,
                    &scales,
                    &mut output,
                    Some(&cached),
                )?;
                let cached_actual = output.to_host_vec(&stream)?;
                if kind == 17 && cols == 5120 {
                    let mut recycled = DeviceBuffer::zeroed(&stream, cached.len() + 256)?;
                    super::fp8_cache::pack_into(
                        &module,
                        &stream,
                        kind,
                        cols,
                        rows,
                        normalizer,
                        &weights,
                        &mut recycled,
                    )?;
                    launch(
                        &module,
                        &stream,
                        kind,
                        cols,
                        rows,
                        batch,
                        normalizer,
                        &weights,
                        &packed,
                        &scales,
                        &mut output,
                        Some(&recycled),
                    )?;
                    assert_eq!(output.to_host_vec(&stream)?, cached_actual);
                }
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    cached_actual
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    "FP8 cached/on-demand mismatch kind{kind} cols{cols} rows{rows} batch{batch}"
                );
                let (mut error, mut energy) = (0.0f64, 0.0f64);
                for row in 0..rows {
                    for token in 0..batch {
                        let (mut expected, mut magnitude, mut reference) = (0.0f64, 0.0f64, 0.0f64);
                        for k in 0..cols {
                            let product = w[row * cols + k] * x[token * cols + k];
                            expected += product;
                            magnitude += product.abs();
                            reference +=
                                original[row * cols + k] as f64 * values[token * cols + k] as f64;
                        }
                        let value = actual[token * rows + row] as f64;
                        anyhow::ensure!(
                            value.is_finite()
                                && (value - expected).abs() <= 3e-5 * magnitude.max(1.0),
                            "FP8 kind{kind} cols{cols} batch{batch} row{row} token{token}: {value} vs{expected}"
                        );
                        error += (value - reference).powi(2);
                        energy += reference * reference;
                    }
                }
                println!(
                    "FP8 kind={kind} cols={cols} batch={batch} original_relative_rms={}",
                    (error / energy.max(1e-30)).sqrt()
                );
            }
        }
        Ok(())
    }
}
