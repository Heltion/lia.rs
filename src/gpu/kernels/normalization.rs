//! Warp reductions for short rows and cooperative blocks for large rows.
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub unsafe fn rms_norm(
        dim: usize,
        groups: usize,
        eps: f32,
        x: &[f32],
        weight: &[f32],
        out: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        if group < groups {
            let mut sum = 0.0f32;
            let mut j = lane;
            while j < dim {
                let v = x[group * dim + j];
                sum += v * v;
                j += 32;
            }
            let mut delta = 16;
            while delta > 0 {
                sum += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum, delta);
                delta /= 2;
            }
            sum = cuda_device::warp::shuffle_f32_sync(0xffff_ffff, sum, 0);
            let scale = 1.0 / (sum / dim as f32 + eps).sqrt();
            j = lane;
            while j < dim {
                unsafe {
                    *out.add(group * dim + j) = x[group * dim + j] * scale * weight[j];
                }
                j += 32;
            }
        }
    }

    #[kernel]
    pub unsafe fn rms_norm_block(
        dim: usize,
        groups: usize,
        eps: f32,
        x: &[f32],
        weight: &[f32],
        out: *mut f32,
    ) {
        static mut SUMS: cuda_device::SharedArray<f32, 8> = cuda_device::SharedArray::UNINIT;
        let tid = thread::index_1d().get() as u32;
        let local = tid % 256;
        let group = tid / 256;
        let lane = local % 32;
        let warp = local / 32;
        let dim = dim as u32;
        if group < groups as u32 {
            let shared = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
            let mut sum = 0.0f32;
            let mut j = local;
            while j < dim {
                let value = unsafe { *x.get_unchecked((group * dim + j) as usize) };
                sum += value * value;
                j += 256;
            }
            let mut step = 16;
            while step > 0 {
                sum += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum, step);
                step /= 2;
            }
            if lane == 0 {
                unsafe {
                    *shared.add(warp as usize) = sum;
                }
            }
            thread::sync_threads();
            if warp == 0 {
                let mut total = if lane < 8 {
                    unsafe { *shared.add(lane as usize) }
                } else {
                    0.0
                };
                step = 16;
                while step > 0 {
                    total += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, total, step);
                    step /= 2;
                }
                if lane == 0 {
                    unsafe {
                        *shared = 1.0 / (total / dim as f32 + eps).sqrt();
                    }
                }
            }
            thread::sync_threads();
            let scale = unsafe { *shared };
            j = local;
            while j < dim {
                unsafe {
                    *out.add((group * dim + j) as usize) = *x
                        .get_unchecked((group * dim + j) as usize)
                        * scale
                        * *weight.get_unchecked(j as usize);
                }
                j += 256;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::gpu::{kernels, tests::close};
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};
    #[test]
    fn block_normalization() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        for dim in [65, 5120] {
            for groups in [1, 3] {
                let input: Vec<f32> = (0..dim * groups)
                    .map(|i| ((i * 17 % 4093) as f32 - 2046.0) * 0.001)
                    .collect();
                let weight: Vec<f32> = (0..dim).map(|i| 0.5 + (i % 13) as f32 * 0.03).collect();
                let x = DeviceBuffer::from_host(&stream, &input)?;
                let w = DeviceBuffer::from_host(&stream, &weight)?;
                let out = DeviceBuffer::<f32>::zeroed(&stream, input.len())?;
                // SAFETY: complete CTAs, matching group and weight extents,
                // including non-warp-aligned tails in each normalization row.
                unsafe {
                    module.normalization.rms_norm_block(
                        &stream,
                        LaunchConfig {
                            grid_dim: (groups as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        dim,
                        groups,
                        1e-6,
                        &x,
                        &w,
                        out.cu_deviceptr() as *mut f32,
                    )?;
                }
                let mut expected = Vec::new();
                for row in input.chunks(dim) {
                    let sum: f64 = row.iter().map(|x| (*x as f64).powi(2)).sum();
                    let scale = 1.0 / (sum / dim as f64 + 1e-6).sqrt();
                    expected.extend(
                        row.iter()
                            .zip(&weight)
                            .map(|(x, w)| *x as f64 * scale * *w as f64),
                    );
                }
                close(
                    &out.to_host_vec(&stream)?,
                    &expected,
                    "block RMS normalization",
                )?;
            }
        }
        Ok(())
    }
}
