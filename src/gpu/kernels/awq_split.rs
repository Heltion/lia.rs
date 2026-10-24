//! Split-K partials use [token][part][row] for the existing F32 reduction.
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub unsafe fn narrow(
        cols: usize,
        rows: usize,
        group_size: usize,
        batch: usize,
        parts: usize,
        weights: &[u32],
        zeros: &[u32],
        scales: &[f32],
        input: &[f32],
        output: *mut f32,
    ) {
        static mut TILE: cuda_device::SharedArray<u32, 2304, 16> = cuda_device::SharedArray::UNINIT;
        unsafe {
            super::super::awq_tile::device::compute::<32, 32, 128, true, false>(
                cols,
                rows,
                group_size,
                batch,
                weights,
                zeros,
                scales,
                input,
                output,
                cuda_device::SharedArray::as_raw_mut_ptr(&raw mut TILE),
                thread::index_1d().get() as u32,
                parts as u32,
            );
        }
    }
    #[kernel]
    pub unsafe fn medium(
        cols: usize,
        rows: usize,
        group_size: usize,
        batch: usize,
        parts: usize,
        weights: &[u32],
        zeros: &[u32],
        scales: &[f32],
        input: &[f32],
        output: *mut f32,
    ) {
        static mut TILE: cuda_device::SharedArray<u32, 2880, 16> = cuda_device::SharedArray::UNINIT;
        unsafe {
            super::super::awq_tile::device::compute::<48, 32, 192, true, false>(
                cols,
                rows,
                group_size,
                batch,
                weights,
                zeros,
                scales,
                input,
                output,
                cuda_device::SharedArray::as_raw_mut_ptr(&raw mut TILE),
                thread::index_1d().get() as u32,
                parts as u32,
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
    fn partitions_match_f64_and_preserve_guards() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        for (rows, cols, batch) in [
            (40usize, 512usize, 9usize),
            (72, 1024, 32),
            (40, 1024, 33),
            (72, 2048, 48),
        ] {
            let group = 128;
            let groups = cols / group;
            let packed: Vec<u32> = (0..rows / 8 * cols)
                .map(|i| (i as u32).wrapping_mul(0x9e3779b9).wrapping_add(0x13579bdf))
                .collect();
            let zeros: Vec<u32> = (0..rows / 8 * groups)
                .map(|i| (i as u32).wrapping_mul(0x13171317).wrapping_add(0x12345678))
                .collect();
            let scales: Vec<f32> = (0..rows * groups)
                .map(|i| (i % 7 + 1) as f32 * 0.00073)
                .collect();
            let x: Vec<f32> = (0..batch * cols)
                .map(|i| ((i * 17 % 257) as f32 - 128.) * 0.0137)
                .collect();
            let dw = DeviceBuffer::from_host(&stream, &packed)?;
            let dz = DeviceBuffer::from_host(&stream, &zeros)?;
            let ds = DeviceBuffer::from_host(&stream, &scales)?;
            let dx = DeviceBuffer::from_host(&stream, &x)?;
            let guard = 12345.0f32;
            let reference = DeviceBuffer::from_host(&stream, &vec![guard; rows * batch + 17])?;
            // SAFETY: complete K64 tiles, 128-value groups and complete CTAs;
            // the output includes guards beyond the logical shape.
            unsafe {
                module.awq_fp16.awq_fp16(
                    &stream,
                    LaunchConfig {
                        grid_dim: ((rows.div_ceil(32) * batch.div_ceil(32)) as u32, 1, 1),
                        block_dim: (128, 1, 1),
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
                    reference.cu_deviceptr() as *mut f32,
                )?;
            }
            let reference = reference.to_host_vec(&stream)?;
            for parts in [2, 4, 8] {
                if !cols.is_multiple_of(parts * group) {
                    continue;
                }
                let partial =
                    DeviceBuffer::from_host(&stream, &vec![guard; rows * batch * parts + 17])?;
                let output = DeviceBuffer::from_host(&stream, &vec![guard; rows * batch + 17])?;
                let medium = batch > 32;
                let tokens = if medium { 48 } else { 32 };
                let config = LaunchConfig {
                    grid_dim: (
                        (rows.div_ceil(32) * batch.div_ceil(tokens) * parts) as u32,
                        1,
                        1,
                    ),
                    block_dim: (if medium { 192 } else { 128 }, 1, 1),
                    shared_mem_bytes: 0,
                };
                // SAFETY: each K partition covers whole groups, writes unique
                // [token][part][row] slots, and all launch extents are bounded.
                unsafe {
                    macro_rules! launch {
                        ($entry:ident) => {
                            module.awq_split.$entry(
                                &stream,
                                config,
                                cols,
                                rows,
                                group,
                                batch,
                                parts,
                                &dw,
                                &dz,
                                &ds,
                                &dx,
                                partial.cu_deviceptr() as *mut f32,
                            )?;
                        };
                    }
                    if medium {
                        launch!(medium);
                    } else {
                        launch!(narrow);
                    }
                    module.awq.awq_merge(
                        &stream,
                        LaunchConfig::for_num_elems((rows * batch) as u32),
                        rows,
                        batch,
                        parts,
                        &partial,
                        output.cu_deviceptr() as *mut f32,
                    )?;
                }
                let actual = output.to_host_vec(&stream)?;
                let partial = partial.to_host_vec(&stream)?;
                ensure!(
                    actual[rows * batch..]
                        .iter()
                        .chain(&partial[rows * batch * parts..])
                        .all(|v| *v == guard),
                    "split output guard overwritten"
                );
                let mut maximum = 0.0f64;
                for t in 0..batch {
                    for r in 0..rows {
                        let nibble = r % 8;
                        let shift = ((nibble % 2) * 4 + nibble / 2) * 4;
                        let expected: f64 = (0..cols)
                            .map(|k| {
                                let code = ((packed[r / 8 * cols + k] >> shift) & 15) as i32;
                                let zero =
                                    ((zeros[r / 8 * groups + k / group] >> shift) & 15) as i32;
                                let w = (((code - zero) as f32 * scales[k / group * rows + r])
                                    as f16) as f32;
                                f64::from(w) * f64::from((x[t * cols + k] as f16) as f32)
                            })
                            .sum();
                        let value = actual[t * rows + r];
                        let error = (f64::from(value) - expected).abs();
                        maximum = maximum.max(error);
                        ensure!(
                            value.is_finite()
                                && error < 5e-5
                                && (value - reference[t * rows + r]).abs() < 5e-5,
                            "split rows={rows} cols={cols} batch={batch} parts={parts} token={t} row={r}: {error}"
                        );
                    }
                }
                println!(
                    "split rows={rows} cols={cols} batch={batch} parts={parts}: f64_error={maximum}, guards intact"
                );
            }
        }
        Ok(())
    }
}
