use super::*;
use anyhow::{Result, ensure};

#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul(
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    bits: u32,
    group_size: usize,
    columns: usize,
    rows: usize,
    packed: &DeviceBuffer<u8>,
    scales: &DeviceBuffer<f32>,
    input: &DeviceBuffer<f32>,
    packed_input: &DeviceBuffer<u32>,
    reuse_input: bool,
    output: &mut DeviceBuffer<f32>,
) -> Result<()> {
    ensure!(
        columns > 0
            && rows > 0
            && columns.is_multiple_of(64)
            && group_size >= 64
            && group_size.is_power_of_two()
            && columns.is_multiple_of(group_size)
            && [2, 4, 8].contains(&bits)
            && input.len().is_multiple_of(columns),
        "invalid GSQ Tensor Core geometry"
    );
    let batch = input.len() / columns;
    ensure!(
        batch > 0
            && batch <= MAX_PREFILL_BATCH
            && rows
                .checked_mul(columns)
                .is_some_and(|n| n <= u32::MAX as usize)
            && rows.checked_mul(batch) == Some(output.len())
            && input.len() <= u32::MAX as usize
            && packed_input.len() >= input.len() / 2
            && packed.len() == rows * columns / (8 / bits as usize)
            && scales.len() == rows * (columns / group_size),
        "invalid GSQ Tensor Core buffer extents"
    );
    // SAFETY: complete pairs, aligned workspaces, complete CTA tiles with row
    // and token tail guards; all buffers remain live on the ordered stream.
    unsafe {
        if !reuse_input {
            module.tensor.pack_input(
                stream,
                LaunchConfig::for_num_elems((input.len() / 2) as u32),
                input,
                packed_input.cu_deviceptr() as *mut u32,
            )?;
        }
        macro_rules! launch {
            ($bits:literal, $tokens:literal) => {
                module.gsq_tensor.matmul::<$bits, $tokens>(
                    stream,
                    LaunchConfig {
                        grid_dim: ((rows.div_ceil(64) * batch.div_ceil($tokens)) as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    columns,
                    rows,
                    batch,
                    group_size.trailing_zeros(),
                    packed,
                    scales,
                    packed_input,
                    output.cu_deviceptr() as *mut f32,
                )?
            };
        }
        match (bits, batch >= 128) {
            (2, false) => launch!(2, 64),
            (2, true) => launch!(2, 128),
            (4, false) => launch!(4, 64),
            (4, true) => launch!(4, 128),
            (8, false) => launch!(8, 64),
            (8, true) => launch!(8, 128),
            _ => unreachable!(),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::gsq::Matrix;

    #[test]
    fn fused_tiles_match_independent_half_operand_reference() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        let (rows, columns) = (67usize, 256usize);
        for (bits, group_size) in [(2, 128), (4, 128), (4, 256), (8, 128), (8, 256)] {
            let matrix = Matrix {
                rows,
                columns,
                bits,
                group_size,
                packed: (0..rows * columns / (8 / bits as usize))
                    .map(|i| i.wrapping_mul(73).wrapping_add(29) as u8)
                    .collect(),
                scales: (0..rows * columns / group_size)
                    .map(|i| (i as i32 % 13 - 6) as f32 / 101.0)
                    .collect(),
            };
            let weights = DeviceBuffer::from_host(&stream, &matrix.packed)?;
            let scales = DeviceBuffer::from_host(&stream, &matrix.scales)?;
            for batch in [1, 9, 65, 129] {
                let input = (0..batch * columns)
                    .map(|i| (i as i32 % 19 - 9) as f32 / 71.0)
                    .collect::<Vec<_>>();
                let dx = DeviceBuffer::from_host(&stream, &input)?;
                let packed = DeviceBuffer::zeroed(&stream, input.len() / 2)?;
                let mut output = DeviceBuffer::zeroed(&stream, batch * rows)?;
                matmul(
                    &module,
                    &stream,
                    bits,
                    group_size,
                    columns,
                    rows,
                    &weights,
                    &scales,
                    &dx,
                    &packed,
                    false,
                    &mut output,
                )?;
                let actual = output.to_host_vec(&stream)?;
                for token in 0..batch {
                    for row in 0..rows {
                        let expected = (0..columns)
                            .map(|c| {
                                let w = (matrix.value(row, c).unwrap() as f16) as f32;
                                let x = (input[token * columns + c] as f16) as f32;
                                w * x
                            })
                            .sum::<f32>();
                        let error = (actual[token * rows + row] - expected).abs();
                        assert!(
                            error < 1e-4 * expected.abs().max(1.0),
                            "bits={bits} group={group_size} batch={batch} token={token} row={row} error={error}"
                        );
                    }
                }
                // Reuse must preserve the same packed immutable activation.
                matmul(
                    &module,
                    &stream,
                    bits,
                    group_size,
                    columns,
                    rows,
                    &weights,
                    &scales,
                    &dx,
                    &packed,
                    true,
                    &mut output,
                )?;
                assert_eq!(actual, output.to_host_vec(&stream)?);
            }
        }
        Ok(())
    }
}
