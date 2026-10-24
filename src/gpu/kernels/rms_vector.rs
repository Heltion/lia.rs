//! Aligned four-float loads shorten the single-row 4096-wide RMS load chain.
use cuda_device::{kernel, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    struct Quad {
        values: [f32; 4],
    }

    #[kernel]
    pub unsafe fn rms_norm_vector4096(eps: f32, x: &[f32], weight: &[f32], out: *mut f32) {
        static mut SUMS: cuda_device::SharedArray<f32, 8> = cuda_device::SharedArray::UNINIT;
        let local = thread::index_1d().get() as u32;
        let lane = local % 32;
        let warp_id = local / 32;
        let shared = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
        let mut sums = [0.0f32; 4];
        let mut j = local * 4;
        while j < 4096 {
            let values = unsafe { x.as_ptr().add(j as usize).cast::<Quad>().read().values };
            let mut i = 0;
            #[unroll]
            while i < 4 {
                sums[i] += values[i] * values[i];
                i += 1;
            }
            j += 1024;
        }
        let mut sum = (sums[0] + sums[1]) + (sums[2] + sums[3]);
        let mut delta = 16;
        while delta > 0 {
            sum += warp::shuffle_down_f32_sync(0xffff_ffff, sum, delta);
            delta /= 2;
        }
        if lane == 0 {
            unsafe {
                *shared.add(warp_id as usize) = sum;
            }
        }
        thread::sync_threads();
        if warp_id == 0 {
            let mut total = if lane < 8 {
                unsafe { *shared.add(lane as usize) }
            } else {
                0.0
            };
            delta = 16;
            while delta > 0 {
                total += warp::shuffle_down_f32_sync(0xffff_ffff, total, delta);
                delta /= 2;
            }
            if lane == 0 {
                unsafe {
                    *shared = 1.0 / (total / 4096.0 + eps).sqrt();
                }
            }
        }
        thread::sync_threads();
        let scale = unsafe { *shared };
        j = local * 4;
        while j < 4096 {
            let values = unsafe { x.as_ptr().add(j as usize).cast::<Quad>().read().values };
            let gains = unsafe { weight.as_ptr().add(j as usize).cast::<Quad>().read().values };
            let mut result = Quad { values: [0.0; 4] };
            let mut i = 0;
            #[unroll]
            while i < 4 {
                result.values[i] = values[i] * scale * gains[i];
                i += 1;
            }
            unsafe {
                out.add(j as usize).cast::<Quad>().write(result);
            }
            j += 1024;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::gpu::{kernels, tests::close};
    use anyhow::{Result, ensure};
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};
    #[test]
    fn vector_norm_matches_f64_and_preserves_guards() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        for magnitude in [0.0f32, 0.001, 1.0, 64.0] {
            for eps in [1e-6f32, 1e-5] {
                let values: Vec<f32> = (0..4096)
                    .map(|i| ((i * 17 % 4093) as f32 - 2046.0) * 0.001 * magnitude)
                    .collect();
                let weights: Vec<f32> = (0..4096).map(|i| 0.5 + (i % 13) as f32 * 0.03).collect();
                let x = DeviceBuffer::from_host(&stream, &values)?;
                let w = DeviceBuffer::from_host(&stream, &weights)?;
                let guard = 12345.0f32;
                let out = DeviceBuffer::from_host(&stream, &vec![guard; 4096 + 16])?;
                // SAFETY: aligned allocation bases, one complete 256-thread CTA,
                // exactly 4096 valid input/weight elements and extra output guards.
                unsafe {
                    module.rms_vector.rms_norm_vector4096(
                        &stream,
                        LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        eps,
                        &x,
                        &w,
                        out.cu_deviceptr() as *mut f32,
                    )?;
                }
                let actual = out.to_host_vec(&stream)?;
                let sum: f64 = values.iter().map(|v| f64::from(*v).powi(2)).sum();
                let scale = 1.0 / (sum / 4096.0 + f64::from(eps)).sqrt();
                let expected: Vec<f64> = values
                    .iter()
                    .zip(&weights)
                    .map(|(x, w)| f64::from(*x) * scale * f64::from(*w))
                    .collect();
                close(&actual[..4096], &expected, "vector RMS4096")?;
                ensure!(
                    actual[4096..].iter().all(|v| *v == guard),
                    "vector RMS output guard overwritten"
                );
            }
        }
        Ok(())
    }
}
