//! Register-fed BF16 MMA validation for the losslessly permuted W2 layout.
use cuda_device::{kernel, thread, wmma};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;

    use super::super::moe_register_math::{coefficients, split};

    #[kernel]
    pub unsafe fn register_mma_projection(
        rows: usize,
        columns: usize,
        weights: &[u16],
        scales: &[u16],
        input: &[f32],
        output: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let tile = tid / 32;
        let lane = tid % 32;
        if tile >= rows / 16 {
            return;
        }
        let mut sum = [0.0; 4];
        let mut k = 0;
        while k < columns {
            let word = weights[(tile * (columns / 16) + k / 16) * 32 + lane] as u32;
            let at = (tile * (columns / 128) + k / 128) * 16 + lane / 4;
            let s = scales[at] as u32;
            let t = scales[at + 8] as u32;
            let a = [
                coefficients(word, s),
                coefficients(word >> 4, t),
                coefficients(word >> 8, s),
                coefficients(word >> 12, t),
            ];
            let column = k + lane % 4 * 2;
            let (b0, l0) = split(input[column], input[column + 1]);
            let (b1, l1) = split(input[column + 8], input[column + 9]);
            // SAFETY: all lanes in a complete warp execute each MMA together.
            unsafe {
                sum = wmma::mma_m16n8k16_f32_bf16(sum, a, [l0, l1]);
                sum = wmma::mma_m16n8k16_f32_bf16(sum, a, [b0, b1]);
            }
            k += 16;
        }
        if lane % 4 == 0 {
            // SAFETY: exactly one lane writes each row in the complete tile.
            unsafe {
                *output.add(tile * 16 + lane / 4) = sum[0];
                *output.add(tile * 16 + lane / 4 + 8) = sum[2];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};

    #[test]
    fn repacked_register_mma_matches_independent_f64_dot_products() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = crate::gpu::kernels::load(&context)?;
        for (rows, columns, exponent) in [
            (16, 128, 3),
            (32, 512, 90),
            (128, 2048, 112),
            (16, 128, 125),
        ] {
            let codes = (0usize..rows * columns / 4)
                .map(|i| ((i * 73 + i / 97) ^ (i >> 5)) as u8)
                .collect::<Vec<_>>();
            let scales = (0..rows * (columns / 128))
                .map(|i| ((exponent << 7) | (i * 17 % 128) | (i % 2 << 15)) as u16)
                .collect::<Vec<_>>();
            let input = (0..columns)
                .map(|i| ((i * 19 % 127) as i32 - 63) as f32 / 33.0)
                .collect::<Vec<_>>();
            let raw = DeviceBuffer::from_host(&stream, &codes)?;
            let raw_scales = DeviceBuffer::from_host(&stream, &scales)?;
            let weights = DeviceBuffer::<u16>::zeroed(&stream, codes.len() / 2)?;
            let reordered = DeviceBuffer::<u16>::zeroed(&stream, scales.len())?;
            let x = DeviceBuffer::from_host(&stream, &input)?;
            let output = DeviceBuffer::<f32>::zeroed(&stream, rows)?;
            // SAFETY: complete tiles, validated normal scales, disjoint buffers;
            // repacking and multiplication are ordered on the same stream.
            unsafe {
                module.moe_repack.weights(
                    &stream,
                    LaunchConfig::for_num_elems(weights.len() as u32),
                    rows,
                    columns,
                    &raw,
                    weights.cu_deviceptr() as *mut u16,
                )?;
                module.moe_repack.scales(
                    &stream,
                    LaunchConfig::for_num_elems(scales.len() as u32),
                    rows,
                    columns,
                    &raw_scales,
                    reordered.cu_deviceptr() as *mut u16,
                )?;
                module.moe_register.register_mma_projection(
                    &stream,
                    LaunchConfig::for_num_elems((rows / 16 * 32) as u32),
                    rows,
                    columns,
                    &weights,
                    &reordered,
                    &x,
                    output.cu_deviceptr() as *mut f32,
                )?;
            }
            let result = output.to_host_vec(&stream)?;
            for row in 0..rows {
                let mut reference = 0.0f64;
                let mut magnitude = 0.0f64;
                for column in 0..columns {
                    let code = (codes[(row * columns + column) / 4] >> (column % 4 * 2)) & 3;
                    let scale =
                        f32::from_bits((scales[row * (columns / 128) + column / 128] as u32) << 16);
                    let product = ((code as i32 - 2) as f32 * scale) as f64 * input[column] as f64;
                    reference += product;
                    magnitude += product.abs();
                }
                let error = (result[row] as f64 - reference).abs();
                assert!(
                    error <= 3e-5 * magnitude + f32::from_bits(1) as f64,
                    "rows={rows}, columns={columns}, row={row}: error={error}, norm={magnitude}"
                );
            }
        }
        Ok(())
    }
}
