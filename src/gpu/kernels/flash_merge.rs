//! Column-parallel attention partition merge with unchanged accumulation order.
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub unsafe fn flash_merge_columns(
        groups: usize,
        parts: usize,
        cap_parts: usize,
        group_offset: usize,
        partial: &[f32],
        qgate: &[f32],
        out: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let group = tid / 256;
        let col = tid % 256;
        if group >= groups {
            return;
        }
        let mut max = f32::NEG_INFINITY;
        let mut p = 0;
        while p < parts {
            max = max.max(partial[(group * cap_parts + p) * 258 + 256]);
            p += 1;
        }
        let mut sum = 0.0f32;
        let mut value = 0.0f32;
        p = 0;
        while p < parts {
            let base = (group * cap_parts + p) * 258;
            let weight = cuda_device::float::ex2_approx_f32(
                (partial[base + 256] - max) * core::f32::consts::LOG2_E,
            );
            sum += weight * partial[base + 257];
            value += weight * partial[base + col];
            p += 1;
        }
        let row = group + group_offset;
        unsafe {
            *out.add(row * 256 + col) = value / sum / (1.0 + (-qgate[row * 512 + 256 + col]).exp());
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};

    #[test]
    fn column_merge_preserves_serial_partition_bits() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = super::super::load(&context)?;
        for (groups, parts, padding, offset) in [
            (16usize, 17usize, 3usize, 0usize),
            (16, 129, 0, 3),
            (7, 256, 2, 1),
        ] {
            let capacity = parts + padding;
            let mut partial = vec![0.0; groups * capacity * 258];
            for g in 0..groups {
                for p in 0..parts {
                    let base = (g * capacity + p) * 258;
                    for col in 0..256 {
                        partial[base + col] =
                            ((g * 53 + p * 71 + col * 37) % 997) as f32 / 71.0 - 7.0;
                    }
                    partial[base + 256] = ((g * 13 + p * 31) % 127) as f32 / 17.0 - 3.0;
                    partial[base + 257] = ((p * 7 + 3) % 257) as f32 / 29.0 + 0.1;
                }
            }
            let gate = (0..(groups + offset) * 512)
                .map(|i| ((i * 73 % 257) as f32 - 128.0) / 101.0)
                .collect::<Vec<_>>();
            let input = DeviceBuffer::from_host(&stream, &partial)?;
            let gate = DeviceBuffer::from_host(&stream, &gate)?;
            let a = DeviceBuffer::<f32>::zeroed(&stream, (groups + offset) * 256)?;
            let b = DeviceBuffer::<f32>::zeroed(&stream, (groups + offset) * 256)?;
            // SAFETY: full 258-value partition rows and disjoint output columns.
            unsafe {
                module.flash_merge.flash_merge_columns(
                    &stream,
                    LaunchConfig {
                        grid_dim: (groups as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    groups,
                    parts,
                    capacity,
                    offset,
                    &input,
                    &gate,
                    a.cu_deviceptr() as *mut f32,
                )?;
                module.attention.flash_merge::<256>(
                    &stream,
                    LaunchConfig::for_num_elems((groups * 32) as u32),
                    groups,
                    parts,
                    capacity,
                    offset,
                    &input,
                    &gate,
                    b.cu_deviceptr() as *mut f32,
                )?;
            }
            let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
            assert_eq!(bits(a.to_host_vec(&stream)?), bits(b.to_host_vec(&stream)?));
        }
        Ok(())
    }
}
