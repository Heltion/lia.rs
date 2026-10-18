//! Row-parallel BF16 routing with the original lane-wise FMA order.
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[kernel]
    pub unsafe fn bf16_router_matvec(
        columns: usize,
        rows: usize,
        batch: usize,
        weights: &[u8],
        input: &[f32],
        output: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        if group >= rows * batch {
            return;
        }
        let row = group % rows;
        let token = group / rows;
        let mut column = lane;
        let mut sum = 0.0f32;
        // Complete 128-column groups permit four independent aligned loads.
        // Their FMAs remain serial in exactly the original col += 32 order.
        while column < columns {
            let mut values = [0.0f32; 4];
            let mut x = [0.0f32; 4];
            let mut i = 0;
            #[unroll]
            while i < 4 {
                let col = column + i * 32;
                let word = unsafe {
                    weights
                        .as_ptr()
                        .cast::<u16>()
                        .add(row * columns + col)
                        .read()
                };
                values[i] = f32::from_bits((word as u32) << 16);
                x[i] = unsafe { *input.get_unchecked(token * columns + col) };
                i += 1;
            }
            i = 0;
            #[unroll]
            while i < 4 {
                sum = values[i].mul_add(x[i], sum);
                i += 1;
            }
            column += 128;
        }
        let mut delta = 16;
        while delta > 0 {
            sum += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum, delta);
            delta /= 2;
        }
        if lane == 0 {
            unsafe {
                *output.add(group) = sum;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};

    #[test]
    fn router_preserves_original_lane_accumulation_bits() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = super::super::load(&context)?;
        for (rows, columns, batch) in [(256usize, 2048usize, 1usize), (7, 128, 3), (256, 512, 8)] {
            let weights = (0..rows * columns)
                .map(|i| {
                    let value = ((i * 73 % 257) as f32 - 128.0) / 101.0;
                    (value.to_bits() >> 16) as u16
                })
                .collect::<Vec<_>>();
            let bytes = weights
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>();
            let x = (0..columns * batch)
                .map(|i| ((i * 37 % 997) as f32 - 498.0) / 71.0)
                .collect::<Vec<_>>();
            let dw = DeviceBuffer::from_host(&stream, &bytes)?;
            let dx = DeviceBuffer::from_host(&stream, &x)?;
            let actual = DeviceBuffer::<f32>::zeroed(&stream, rows * batch)?;
            let reference = DeviceBuffer::<f32>::zeroed(&stream, rows * batch)?;
            // SAFETY: complete BF16 rows, full warps, and 128-column groups.
            unsafe {
                module.bf16_router.bf16_router_matvec(
                    &stream,
                    LaunchConfig {
                        grid_dim: ((rows * batch * 32).div_ceil(64) as u32, 1, 1),
                        block_dim: (64, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    columns,
                    rows,
                    batch,
                    &dw,
                    &dx,
                    actual.cu_deviceptr() as *mut f32,
                )?;
                if batch == 1 {
                    module.matrix.matvec_packed::<30, 2, 1>(
                        &stream,
                        LaunchConfig::for_num_elems((rows * 32) as u32),
                        columns,
                        rows,
                        &dw,
                        &dx,
                        reference.cu_deviceptr() as *mut f32,
                    )?;
                } else {
                    module.matrix.matmul_eight::<30, 2, 1>(
                        &stream,
                        LaunchConfig::for_num_elems((rows * 32) as u32),
                        columns,
                        rows,
                        batch,
                        &dw,
                        &dx,
                        reference.cu_deviceptr() as *mut f32,
                    )?;
                }
            }
            let actual = actual.to_host_vec(&stream)?;
            let reference = reference.to_host_vec(&stream)?;
            for (i, (a, b)) in actual.iter().zip(&reference).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "rows={rows} cols={columns} batch={batch} index={i}"
                );
            }
        }
        Ok(())
    }
}
