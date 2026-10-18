//! Checked FP8 weight tile conversion into a fixed reusable working buffer.
use super::*;

pub(crate) const MIN_BATCH: usize = 512;

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn pack(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    kind: u32,
    cols: usize,
    rows: usize,
    normalizer: f32,
    weights: &DeviceBuffer<u8>,
) -> anyhow::Result<DeviceBuffer<u32>> {
    let padded = rows
        .div_ceil(64)
        .checked_mul(64)
        .and_then(|r| r.checked_mul(cols))
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or_else(|| anyhow!("FP8 cache shape overflow"))?;
    let mut output = DeviceBuffer::zeroed(stream, padded / 4)?;
    pack_into(
        module,
        stream,
        kind,
        cols,
        rows,
        normalizer,
        weights,
        &mut output,
    )?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn pack_into(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    kind: u32,
    cols: usize,
    rows: usize,
    normalizer: f32,
    weights: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<u32>,
) -> anyhow::Result<()> {
    let (block, bytes) =
        crate::gguf::layout(kind).ok_or_else(|| anyhow!("unknown FP8 cache format"))?;
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| anyhow!("FP8 cache shape overflow"))?;
    anyhow::ensure!(
        cols > 0
            && cols.is_multiple_of(256)
            && rows > 0
            && elements <= u32::MAX as usize
            && normalizer.is_finite()
            && normalizer > 0.0
            && weights.len() == elements / block as usize * bytes as usize,
        "FP8 cache shape mismatch"
    );
    let padded = rows
        .div_ceil(64)
        .checked_mul(64)
        .and_then(|r| r.checked_mul(cols))
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or_else(|| anyhow!("FP8 cache padded shape overflow"))?;
    anyhow::ensure!(
        output.len() >= padded / 4,
        "FP8 weight tile workspace too small"
    );
    let config = LaunchConfig::for_num_elems((padded / 8) as u32);
    macro_rules! pack {
        ($kind:literal, $bytes:literal) => {
            module.fp8_weights.fp8_pack_weight_tiles::<$kind, $bytes>(
                stream,
                config,
                cols,
                rows,
                padded,
                normalizer,
                weights,
                &mut *output,
            )?
        };
    }
    // SAFETY: bounded complete quantization blocks and uniquely owned padded output.
    unsafe {
        match kind {
            10 => pack!(10, 84),
            12 => pack!(12, 144),
            14 => pack!(14, 210),
            16 => pack!(16, 66),
            17 => pack!(17, 74),
            18 => pack!(18, 98),
            19 => pack!(19, 50),
            21 => pack!(21, 110),
            22 => pack!(22, 82),
            23 => pack!(23, 136),
            29 => pack!(29, 56),
            _ => return Err(anyhow!("unsupported FP8 cache format {kind}")),
        }
    }
    Ok(())
}
