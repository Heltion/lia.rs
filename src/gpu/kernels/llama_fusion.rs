//! Packing fuses with FP32 normalization and activation without changing arithmetic.
use cuda_device::{kernel, thread, warp};
use cuda_host::cuda_module;
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub unsafe fn norm_pack<const ADD: bool>(
        groups: usize,
        epsilon: f32,
        input: &[f32],
        add: &[f32],
        weight: &[f32],
        residual: *mut f32,
        packed: *mut u32,
    ) {
        let tid = thread::index_1d().get();
        let local = tid % 256;
        let group = tid / 256;
        let lane = local % 32;
        let warp_id = local / 32;
        static mut SUMS: cuda_device::SharedArray<f32, 8> = cuda_device::SharedArray::UNINIT;
        let shared = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
        if group < groups {
            let mut values = [0_f32; 8];
            let mut sum = 0_f32;
            let mut i = 0;
            #[unroll]
            while i < 8 {
                let at = group * 2048 + local + i * 256;
                let value = if ADD { input[at] + add[at] } else { input[at] };
                values[i] = value;
                sum += value * value;
                if ADD {
                    unsafe {
                        *residual.add(at) = value;
                    }
                }
                i += 1;
            }
            let mut step = 16;
            while step > 0 {
                sum += warp::shuffle_down_f32_sync(0xffff_ffff, sum, step);
                step /= 2;
            }
            if lane == 0 {
                unsafe {
                    *shared.add(warp_id) = sum;
                }
            }
            thread::sync_threads();
            if warp_id == 0 {
                let mut total = if lane < 8 {
                    unsafe { *shared.add(lane) }
                } else {
                    0.0
                };
                step = 16;
                while step > 0 {
                    total += warp::shuffle_down_f32_sync(0xffff_ffff, total, step);
                    step /= 2;
                }
                if lane == 0 {
                    unsafe {
                        *shared = 1.0 / (total / 2048.0 + epsilon).sqrt();
                    }
                }
            }
            thread::sync_threads();
            let scale = unsafe { *shared };
            i = 0;
            #[unroll]
            while i < 8 {
                let column = local + i * 256;
                let value = values[i] * scale * weight[column];
                let next = warp::shuffle_down_f32_sync(0xffff_ffff, value, 1);
                if lane.is_multiple_of(2) {
                    unsafe {
                        *packed.add((group * 2048 + column) / 2) =
                            cuda_device::convert::cvt_f16x2_f32(value, next);
                    }
                }
                i += 1;
            }
        }
    }

    #[kernel]
    pub unsafe fn silu_pack(gate: &[f32], up: &[f32], packed: *mut u32) {
        let i = thread::index_1d().get() * 2;
        if i < gate.len() {
            let a = up[i] * gate[i] / (1.0 + (-gate[i]).exp());
            let b = up[i + 1] * gate[i + 1] / (1.0 + (-gate[i + 1]).exp());
            unsafe {
                *packed.add(i / 2) = cuda_device::convert::cvt_f16x2_f32(a, b);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::gpu::kernels;
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};
    #[test]
    #[ignore = "requires CUDA"]
    fn packed_normalization_residual_and_silu_match_existing_operators() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        let groups = 19;
        let input: Vec<_> = (0..groups * 2048)
            .map(|i| (i % 67) as f32 / 32.0 - 1.0)
            .collect();
        let add: Vec<_> = (0..input.len())
            .map(|i| (i % 43) as f32 / 64.0 - 0.25)
            .collect();
        let weight: Vec<_> = (0..2048).map(|i| 0.75 + (i % 17) as f32 / 32.0).collect();
        let x = DeviceBuffer::from_host(&stream, &input)?;
        let y = DeviceBuffer::from_host(&stream, &add)?;
        let w = DeviceBuffer::from_host(&stream, &weight)?;
        let mut normalized = DeviceBuffer::<f32>::zeroed(&stream, input.len())?;
        let residual = DeviceBuffer::<f32>::zeroed(&stream, input.len())?;
        let packed = DeviceBuffer::<u32>::zeroed(&stream, input.len() / 2)?;
        let reference = DeviceBuffer::<u32>::zeroed(&stream, input.len() / 2)?;
        let config = LaunchConfig {
            grid_dim: (groups as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        for add_values in [false, true] {
            let expected: Vec<_> = input
                .iter()
                .zip(&add)
                .map(|(&a, &b)| if add_values { a + b } else { a })
                .collect();
            let source = DeviceBuffer::from_host(&stream, &expected)?;
            unsafe {
                module.normalization.rms_norm_block(
                    &stream,
                    config,
                    2048,
                    groups,
                    1e-6,
                    &source,
                    &w,
                    normalized.cu_deviceptr() as *mut f32,
                )?;
                module.tensor.pack_input(
                    &stream,
                    LaunchConfig::for_num_elems((input.len() / 2) as u32),
                    &normalized,
                    reference.cu_deviceptr() as *mut u32,
                )?;
                if add_values {
                    module.llama_fusion.norm_pack::<true>(
                        &stream,
                        config,
                        groups,
                        1e-6,
                        &x,
                        &y,
                        &w,
                        residual.cu_deviceptr() as *mut f32,
                        packed.cu_deviceptr() as *mut u32,
                    )?;
                } else {
                    module.llama_fusion.norm_pack::<false>(
                        &stream,
                        config,
                        groups,
                        1e-6,
                        &x,
                        &y,
                        &w,
                        residual.cu_deviceptr() as *mut f32,
                        packed.cu_deviceptr() as *mut u32,
                    )?;
                }
            }
            assert_eq!(
                packed.to_host_vec(&stream)?,
                reference.to_host_vec(&stream)?,
                "packed normalization differs"
            );
            if add_values {
                assert_eq!(residual.to_host_vec(&stream)?, expected);
            }
        }
        let original_up = y.to_host_vec(&stream)?;
        normalized.copy_from_host(&stream, &input)?;
        unsafe {
            module.basic.silu_mul_in_place(
                &stream,
                LaunchConfig::for_num_elems(input.len() as u32),
                &y,
                &mut normalized,
            )?;
            module.tensor.pack_input(
                &stream,
                LaunchConfig::for_num_elems((input.len() / 2) as u32),
                &normalized,
                reference.cu_deviceptr() as *mut u32,
            )?;
            module.llama_fusion.silu_pack(
                &stream,
                LaunchConfig::for_num_elems((input.len() / 2) as u32),
                &y,
                &x,
                packed.cu_deviceptr() as *mut u32,
            )?;
        }
        assert_eq!(
            packed.to_host_vec(&stream)?,
            reference.to_host_vec(&stream)?,
            "packed SiLU differs"
        );
        assert_eq!(y.to_host_vec(&stream)?, original_up);
        Ok(())
    }
}
