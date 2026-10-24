//! AWQ tiles reuse each packed word across 32 or 64 input tokens.
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub unsafe fn awq_fp16(
        cols: usize,
        rows: usize,
        group_size: usize,
        batch: usize,
        weights: &[u32],
        zeros: &[u32],
        scales: &[f32],
        input: &[f32],
        output: *mut f32,
    ) {
        static mut TILE: cuda_device::SharedArray<u32, 2304, 16> = cuda_device::SharedArray::UNINIT;
        unsafe {
            let tile = cuda_device::SharedArray::as_raw_mut_ptr(&raw mut TILE);
            super::super::awq_tile::device::compute::<32, 32, 128, false, false>(
                cols,
                rows,
                group_size,
                batch,
                weights,
                zeros,
                scales,
                input,
                output,
                tile,
                thread::index_1d().get() as u32,
                1,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::gpu::kernels;
    use anyhow::{Result, ensure};
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};

    #[test]
    fn packed_tiles_cover_row_token_and_group_boundaries() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        for (rows, cols, batch, group) in [
            (8usize, 64usize, 1usize, 64usize),
            (24, 128, 9, 128),
            (40, 192, 33, 64),
            (72, 256, 65, 128),
            (64, 128, 63, 128),
            (64, 128, 64, 128),
            (40, 192, 129, 64),
            (72, 256, 512, 128),
            (64, 128, 32, 128),
            (40, 192, 47, 64),
            (72, 256, 48, 128),
            (40, 128, 49, 128),
        ] {
            let groups = cols / group;
            let packed: Vec<u32> = (0..rows / 8 * cols)
                .map(|i| (i as u32).wrapping_mul(0x9e3779b9).wrapping_add(0x13579bdf))
                .collect();
            let zeros: Vec<u32> = (0..rows / 8 * groups)
                .map(|i| (i as u32).wrapping_mul(0x13171317).wrapping_add(0x12345678))
                .collect();
            let scales: Vec<f32> = (0..rows * groups)
                .map(|i| (i % 7 + 1) as f32 * 0.0073)
                .collect();
            let x: Vec<f32> = (0..batch * cols)
                .map(|i| ((i * 17 % 257) as f32 - 128.) * 0.0137)
                .collect();
            let dw = DeviceBuffer::from_host(&stream, &packed)?;
            let dz = DeviceBuffer::from_host(&stream, &zeros)?;
            let ds = DeviceBuffer::from_host(&stream, &scales)?;
            let dx = DeviceBuffer::from_host(&stream, &x)?;
            let guard = 12345.0f32;
            let output = DeviceBuffer::from_host(&stream, &vec![guard; rows * batch + 17])?;
            let reference = DeviceBuffer::from_host(&stream, &vec![guard; rows * batch + 17])?;
            let wide = DeviceBuffer::from_host(&stream, &vec![guard; rows * batch + 17])?;
            let square = DeviceBuffer::from_host(&stream, &vec![guard; rows * batch + 17])?;
            let medium = DeviceBuffer::from_host(&stream, &vec![guard; rows * batch + 17])?;
            let config = LaunchConfig {
                grid_dim: ((rows.div_ceil(32) * batch.div_ceil(32)) as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: complete64-column tiles and groups, aligned packed words,
            // complete four-, six- or eight-warp CTAs and guarded output extents.
            unsafe {
                module.awq_fp16.awq_fp16(
                    &stream,
                    config,
                    cols,
                    rows,
                    group,
                    batch,
                    &dw,
                    &dz,
                    &ds,
                    &dx,
                    output.cu_deviceptr() as *mut f32,
                )?;
                module.awq_tile.awq_fp16_wide(
                    &stream,
                    LaunchConfig {
                        grid_dim: ((rows.div_ceil(32) * batch.div_ceil(64)) as u32, 1, 1),
                        ..config
                    },
                    cols,
                    rows,
                    group,
                    batch,
                    &dw,
                    &dz,
                    &ds,
                    &dx,
                    wide.cu_deviceptr() as *mut f32,
                )?;
                module.awq_tile.awq_fp16_square(
                    &stream,
                    LaunchConfig {
                        grid_dim: ((rows.div_ceil(64) * batch.div_ceil(64)) as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    cols,
                    rows,
                    group,
                    batch,
                    &dw,
                    &dz,
                    &ds,
                    &dx,
                    square.cu_deviceptr() as *mut f32,
                )?;
                module.awq_medium.awq_fp16_medium(
                    &stream,
                    LaunchConfig {
                        grid_dim: ((rows.div_ceil(32) * batch.div_ceil(48)) as u32, 1, 1),
                        block_dim: (192, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    cols,
                    rows,
                    group,
                    batch,
                    &dw,
                    &dz,
                    &ds,
                    &dx,
                    medium.cu_deviceptr() as *mut f32,
                )?;
                module.awq_matrix.awq_matrix::<false>(
                    &stream,
                    config,
                    cols,
                    rows,
                    group,
                    batch,
                    &dw,
                    &dz,
                    &ds,
                    &dx,
                    reference.cu_deviceptr() as *mut f32,
                )?;
            }
            let a = output.to_host_vec(&stream)?;
            let b = reference.to_host_vec(&stream)?;
            let c = wide.to_host_vec(&stream)?;
            let d = square.to_host_vec(&stream)?;
            let e = medium.to_host_vec(&stream)?;
            ensure!(
                a.iter().zip(&e).all(|(a, b)| a.to_bits() == b.to_bits()),
                "N48 tile changed FP16 product bits or guards"
            );
            ensure!(
                a.iter().zip(&d).all(|(a, d)| a.to_bits() == d.to_bits()),
                "M64N64 tile changed FP16 product bits or guards"
            );
            ensure!(
                a.iter().zip(&c).all(|(a, c)| a.to_bits() == c.to_bits()),
                "N64 tile changed FP16 product bits or output guards"
            );
            ensure!(
                a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()),
                "K64 tile changed FP16 product bits"
            );
            ensure!(
                a[rows * batch..]
                    .iter()
                    .chain(&b[rows * batch..])
                    .all(|x| *x == guard),
                "tile output guard overwritten"
            );
            let mut maximum = 0.0f64;
            for t in 0..batch {
                for r in 0..rows {
                    let nibble = r % 8;
                    let shift = ((nibble % 2) * 4 + nibble / 2) * 4;
                    let expected: f64 = (0..cols)
                        .map(|k| {
                            let code = ((packed[r / 8 * cols + k] >> shift) & 15) as i32;
                            let zero = ((zeros[r / 8 * groups + k / group] >> shift) & 15) as i32;
                            let w = (((code - zero) as f32 * scales[k / group * rows + r]) as f16)
                                as f32;
                            let v = (x[t * cols + k] as f16) as f32;
                            f64::from(w) * f64::from(v)
                        })
                        .sum();
                    let error = (f64::from(a[t * rows + r]) - expected).abs();
                    maximum = maximum.max(error);
                    ensure!(
                        a[t * rows + r].is_finite() && error < 0.00005,
                        "tile f64 mismatch: {error}"
                    );
                }
            }
            println!(
                "K64 rows={rows} cols={cols} batch={batch} group={group}: bit-exact, f64_error={maximum}, guards intact"
            );
        }
        Ok(())
    }
}
