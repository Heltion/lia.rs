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
    output: &mut DeviceBuffer<f32>,
) -> Result<()> {
    ensure!(
        columns > 0
            && rows > 0
            && [2, 4, 8].contains(&bits)
            && group_size.is_power_of_two()
            && columns.is_multiple_of(group_size)
            && columns.is_multiple_of(32 / bits as usize),
        "invalid GSQ matrix geometry"
    );
    ensure!(
        input.len().is_multiple_of(columns),
        "incomplete GSQ input rows"
    );
    let batch = input.len() / columns;
    ensure!(
        batch > 0
            && output.len()
                == rows
                    .checked_mul(batch)
                    .ok_or_else(|| anyhow!("GSQ output overflow"))?,
        "invalid GSQ output shape"
    );
    ensure!(
        packed.len()
            == rows
                .checked_mul(columns)
                .and_then(|n| n.checked_div(8 / bits as usize))
                .ok_or_else(|| anyhow!("GSQ weight extent overflow"))?
            && scales.len() == rows * (columns / group_size),
        "invalid GSQ weight or scale extent"
    );
    let threads = output
        .len()
        .checked_mul(32)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| anyhow!("GSQ launch extent overflow"))?;
    // SAFETY: all extents and complete warp geometry are checked above. One
    // warp owns each output, and all data resides on the supplied stream.
    unsafe {
        macro_rules! vector {
            ($bits:literal, $channel:literal) => {
                module.gsq.matvec4::<$bits, $channel>(
                    stream,
                    LaunchConfig::for_num_elems(threads),
                    columns,
                    rows,
                    batch,
                    group_size.trailing_zeros(),
                    packed,
                    scales,
                    input,
                    output.cu_deviceptr() as *mut f32,
                )?
            };
        }
        macro_rules! scalar {
            ($bits:literal) => {
                module.gsq.matmul::<$bits>(
                    stream,
                    LaunchConfig::for_num_elems(threads),
                    columns,
                    rows,
                    batch,
                    group_size.trailing_zeros(),
                    packed,
                    scales,
                    input,
                    output.cu_deviceptr() as *mut f32,
                )?
            };
        }
        if columns.is_multiple_of(128) && group_size >= 128 {
            match (bits, group_size == columns) {
                (2, false) => vector!(2, false),
                (2, true) => vector!(2, true),
                (4, false) => vector!(4, false),
                (4, true) => vector!(4, true),
                (8, false) => vector!(8, false),
                (8, true) => vector!(8, true),
                _ => unreachable!(),
            }
        } else if bits == 2 {
            scalar!(2);
        } else if bits == 4 {
            scalar!(4);
        } else {
            scalar!(8);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::gsq::Matrix;

    #[test]
    fn grouped_matmul_matches_independent_scalar_decoding() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        for (bits, group_size) in [(2, 128), (2, 256), (4, 128), (4, 256), (8, 128), (8, 256)] {
            let (rows, columns, batch) = (35, 256, 3);
            let packed = (0..rows * columns / (8 / bits as usize))
                .map(|i| (i.wrapping_mul(73).wrapping_add(29)) as u8)
                .collect::<Vec<_>>();
            let scales = (0..rows * columns / group_size)
                .map(|i| (i as i32 % 17 - 8) as f32 / 128.0)
                .collect::<Vec<_>>();
            let input = (0..columns * batch)
                .map(|i| (i % 19) as f32 / 17.0 - 0.5)
                .collect::<Vec<_>>();
            let matrix = Matrix {
                rows,
                columns,
                bits,
                group_size,
                packed,
                scales,
            };
            let dp = DeviceBuffer::from_host(&stream, &matrix.packed)?;
            let ds = DeviceBuffer::from_host(&stream, &matrix.scales)?;
            let dx = DeviceBuffer::from_host(&stream, &input)?;
            let mut dy = DeviceBuffer::zeroed(&stream, rows * batch)?;
            matmul(
                &module, &stream, bits, group_size, columns, rows, &dp, &ds, &dx, &mut dy,
            )?;
            let actual = dy.to_host_vec(&stream)?;
            for token in 0..batch {
                for row in 0..rows {
                    let expected = (0..columns)
                        .map(|c| matrix.value(row, c).unwrap() * input[token * columns + c])
                        .sum::<f32>();
                    let error = (actual[token * rows + row] - expected).abs();
                    assert!(
                        error < 0.0001 * expected.abs().max(1.0),
                        "bits={bits}, token={token}, row={row}, error={error}"
                    );
                }
            }
        }
        Ok(())
    }
}
