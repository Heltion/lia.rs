//! Joint launches for two narrow BF16 projections, preserving each reduction.
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub unsafe fn bf16_pair_partial(
        cols: usize,
        rows: usize,
        batch: usize,
        parts: usize,
        first: &[u8],
        second: &[u8],
        x: &[f32],
        out: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        let count = rows * batch * parts;
        if group < count * 2 {
            let weights = if group < count { first } else { second };
            let local = group % count;
            let part = local % parts;
            let row = (local / parts) % rows;
            let token = local / (parts * rows);
            let mut col = part * 256 + lane;
            let mut sum = 0.0f32;
            while col < (part + 1) * 256 && col < cols {
                let off = 2 * (row * cols + col);
                let bits = weights[off] as u32 | ((weights[off + 1] as u32) << 8);
                let w = f32::from_bits(bits << 16);
                sum = w.mul_add(x[token * cols + col], sum);
                col += 32;
            }
            let mut delta = 16;
            while delta > 0 {
                sum += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum, delta);
                delta /= 2;
            }
            if lane == 0 {
                unsafe {
                    *out.add(group) = sum;
                }
            }
        }
    }

    #[kernel]
    pub unsafe fn bf16_pair_merge(parts: usize, input: &[f32], first: *mut f32, second: *mut f32) {
        let tid = thread::index_1d().get();
        let row = tid / 32;
        let lane = tid % 32;
        if row < input.len() / parts {
            let mut sum = 0.0f32;
            let mut i = lane;
            while i < parts {
                sum += input[row * parts + i];
                i += 32;
            }
            let mut delta = 16;
            while delta > 0 {
                sum += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum, delta);
                delta /= 2;
            }
            if lane == 0 {
                unsafe {
                    let count = input.len() / (2 * parts);
                    let out = if row < count { first } else { second };
                    *out.add(row % count) = sum;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};

    #[test]
    fn paired_projections_preserve_independent_reduction_bits() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = super::super::load(&context)?;
        for (cols, rows, batch) in [(2048usize, 32usize, 1usize), (544, 7, 3), (128, 1, 8)] {
            let weights = (0..2)
                .map(|matrix| {
                    (0..cols * rows)
                        .flat_map(|i| {
                            let value = ((i * 73 + matrix * 19) % 257) as f32 / 101.0 - 1.0;
                            ((value.to_bits() >> 16) as u16).to_le_bytes()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let x = (0..cols * batch)
                .map(|i| ((i * 37 % 997) as f32 - 498.0) / 71.0)
                .collect::<Vec<_>>();
            let first = DeviceBuffer::from_host(&stream, &weights[0])?;
            let second = DeviceBuffer::from_host(&stream, &weights[1])?;
            let input = DeviceBuffer::from_host(&stream, &x)?;
            let parts = cols.div_ceil(256);
            let count = rows * batch;
            let partial = DeviceBuffer::<f32>::zeroed(&stream, 2 * parts * count)?;
            let a = DeviceBuffer::<f32>::zeroed(&stream, count)?;
            let b = DeviceBuffer::<f32>::zeroed(&stream, count)?;
            // SAFETY: complete BF16 matrices and disjoint paired outputs.
            unsafe {
                module.bf16_pair.bf16_pair_partial(
                    &stream,
                    LaunchConfig::for_num_elems((2 * parts * count * 32) as u32),
                    cols,
                    rows,
                    batch,
                    parts,
                    &first,
                    &second,
                    &input,
                    partial.cu_deviceptr() as *mut f32,
                )?;
                module.bf16_pair.bf16_pair_merge(
                    &stream,
                    LaunchConfig::for_num_elems((2 * count * 32) as u32),
                    parts,
                    &partial,
                    a.cu_deviceptr() as *mut f32,
                    b.cu_deviceptr() as *mut f32,
                )?;
            }
            for (weight, actual) in [(&first, &a), (&second, &b)] {
                let independent = DeviceBuffer::<f32>::zeroed(&stream, parts * count)?;
                let expected = DeviceBuffer::<f32>::zeroed(&stream, count)?;
                // SAFETY: same complete dimensions as the paired launch.
                unsafe {
                    module.matrix.bf16_partial(
                        &stream,
                        LaunchConfig::for_num_elems((parts * count * 32) as u32),
                        cols,
                        rows,
                        batch,
                        parts,
                        weight,
                        &input,
                        independent.cu_deviceptr() as *mut f32,
                    )?;
                    module.matrix.bf16_merge(
                        &stream,
                        LaunchConfig::for_num_elems((count * 32) as u32),
                        parts,
                        &independent,
                        expected.cu_deviceptr() as *mut f32,
                    )?;
                }
                let bits =
                    |values: Vec<f32>| values.into_iter().map(f32::to_bits).collect::<Vec<_>>();
                assert_eq!(
                    bits(actual.to_host_vec(&stream)?),
                    bits(expected.to_host_vec(&stream)?)
                );
            }
        }
        Ok(())
    }
}
