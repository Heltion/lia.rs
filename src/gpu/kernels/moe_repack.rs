//! Lossless W2 and BF16 scale permutation into warp-distributed MMA fragments.
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[inline(always)]
    unsafe fn pair(input: &[u8], columns: usize, row: usize, column: usize) -> u16 {
        let index = row * columns + column;
        // SAFETY: the caller supplies complete 16-row, 128-column tiles. An
        // even column keeps both two-bit coefficients in this byte.
        ((unsafe { *input.get_unchecked(index / 4) } >> ((column % 4) * 2)) & 15) as u16
    }

    #[kernel]
    pub unsafe fn weights(rows: usize, columns: usize, input: &[u8], output: *mut u16) {
        let i = thread::index_1d().get();
        if i >= rows * columns / 8 {
            return;
        }
        let lane = i % 32;
        let tile = i / 32;
        let row = tile / (columns / 16) * 16 + lane / 4;
        let column = tile % (columns / 16) * 16 + lane % 4 * 2;
        // The four nibbles follow the A-register order of m16n8k16. Each
        // output word owns eight distinct coefficients of a 16-by-16 tile.
        unsafe {
            *output.add(i) = pair(input, columns, row, column)
                | pair(input, columns, row + 8, column) << 4
                | pair(input, columns, row, column + 8) << 8
                | pair(input, columns, row + 8, column + 8) << 12;
        }
    }

    #[kernel]
    pub unsafe fn scales(rows: usize, columns: usize, input: &[u16], output: *mut u16) {
        let i = thread::index_1d().get();
        let groups = columns / 128;
        if i < rows * groups {
            let row = i / (groups * 16) * 16 + i % 16;
            let group = i / 16 % groups;
            // SAFETY: this is a bijection of complete sixteen-row groups;
            // scale bits are copied without any floating-point conversion.
            unsafe { *output.add(i) = *input.get_unchecked(row * groups + group) };
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};

    #[test]
    fn mma_repack_preserves_every_coefficient_and_scale_bit() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = crate::gpu::kernels::load(&context)?;
        for (rows, columns) in [(16, 128), (128, 512), (512, 2048), (32768, 128)] {
            let input = (0usize..rows * columns / 4)
                .map(|i| ((i * 73 + i / 97) ^ (i >> 5)) as u8)
                .collect::<Vec<_>>();
            let scales = (0..rows * (columns / 128))
                .map(|i| (i * 167 + i / 11) as u16)
                .collect::<Vec<_>>();
            let weights_gpu = DeviceBuffer::from_host(&stream, &input)?;
            let scales_gpu = DeviceBuffer::from_host(&stream, &scales)?;
            let packed = DeviceBuffer::<u16>::zeroed(&stream, input.len() / 2)?;
            let reordered = DeviceBuffer::<u16>::zeroed(&stream, scales.len())?;
            // SAFETY: complete 16-row/128-column tiles, exact immutable input
            // extents, and disjoint equally sized output allocations.
            unsafe {
                module.moe_repack.weights(
                    &stream,
                    LaunchConfig::for_num_elems((input.len() / 2) as u32),
                    rows,
                    columns,
                    &weights_gpu,
                    packed.cu_deviceptr() as *mut u16,
                )?;
                module.moe_repack.scales(
                    &stream,
                    LaunchConfig::for_num_elems(scales.len() as u32),
                    rows,
                    columns,
                    &scales_gpu,
                    reordered.cu_deviceptr() as *mut u16,
                )?;
            }
            let packed = packed.to_host_vec(&stream)?;
            let reordered = reordered.to_host_vec(&stream)?;
            assert_eq!(packed.len() * 2, input.len());
            for row in 0..rows {
                // Decode the new layout back into logical matrix coordinates,
                // rather than repeating the GPU's output-word construction.
                for column in 0..columns {
                    let lane = row % 8 * 4 + column % 8 / 2;
                    let fragment = column % 16 / 8 * 2 + row % 16 / 8;
                    let i = (row / 16 * (columns / 16) + column / 16) * 32 + lane;
                    let actual = (packed[i] >> (fragment * 4 + column % 2 * 2)) & 3;
                    let linear = row * columns + column;
                    let expected = (input[linear / 4] >> (column % 4 * 2)) & 3;
                    assert_eq!(actual as u8, expected, "row={row},column={column}");
                }
                for group in 0..columns / 128 {
                    let i = (row / 16 * (columns / 128) + group) * 16 + row % 16;
                    assert_eq!(reordered[i], scales[row * (columns / 128) + group]);
                }
            }
        }
        Ok(())
    }
}
