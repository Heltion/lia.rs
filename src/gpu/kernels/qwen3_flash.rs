//! Tiled 32-query/8-KV-head attention: tensor QK and FP32 probability/value sums.
use cuda_device::{kernel, launch_bounds};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    #[launch_bounds(128, 3)]
    pub unsafe fn qwen3_flash(
        batch: usize,
        position: usize,
        q: &[f32],
        k: &[u16],
        v: &[u16],
        output: *mut f32,
    ) {
        unsafe {
            super::super::llama_flash_kernel::attention::<64, 128, 32, 8, 4, true, false, false>(
                0,
                batch,
                position,
                1,
                position + batch,
                false,
                0,
                &[],
                &[],
                &[],
                q,
                k,
                v,
                true,
                output,
            );
        }
    }
    #[kernel]
    #[launch_bounds(128, 3)]
    pub unsafe fn qwen3_flash_residual(
        batch: usize,
        position: usize,
        q: &[f32],
        k: &[u16],
        v: &[u16],
        output: *mut f32,
    ) {
        unsafe {
            super::super::llama_flash_kernel::attention::<64, 128, 32, 8, 4, true, true, false>(
                0,
                batch,
                position,
                1,
                position + batch,
                false,
                0,
                &[],
                &[],
                &[],
                q,
                k,
                v,
                true,
                output,
            );
        }
    }
    #[kernel]
    #[launch_bounds(128, 3)]
    pub unsafe fn qwen3_flash_register(
        batch: usize,
        position: usize,
        q: &[f32],
        k: &[u16],
        v: &[u16],
        output: *mut f32,
    ) {
        unsafe {
            super::super::llama_flash_kernel::attention::<64, 128, 32, 8, 4, true, true, true>(
                0,
                batch,
                position,
                1,
                position + batch,
                false,
                0,
                &[],
                &[],
                &[],
                q,
                k,
                v,
                true,
                output,
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
    fn gqa_and_causal_query_tails_match_scalar_and_f64() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        for (position, batch, gain) in [
            (0usize, 1usize, 0usize),
            (0, 17, 0),
            (31, 33, 0),
            (129, 65, 0),
            (1000, 97, 0),
            (4090, 33, 0),
            (31, 17, 1),
            (129, 33, 2),
            (1000, 17, 4),
        ] {
            let tokens = position + batch;
            let q: Vec<f32> = (0..batch * 4096)
                .map(|i| {
                    if gain > 0 {
                        gain as f32
                    } else {
                        ((i * 17 % 257) as f32 - 128.) * 0.0037
                    }
                })
                .collect();
            let k: Vec<f16> = (0..(tokens + 11) * 1024)
                .map(|i| {
                    if i / 1024 >= tokens {
                        10.0 as f16
                    } else if gain > 0 {
                        (((i / 1024 % 65) as f32 - 64.) * 0.0625) as f16
                    } else {
                        (((i * 23 % 263) as f32 - 131.) * 0.0043) as f16
                    }
                })
                .collect();
            let v: Vec<f16> = (0..k.len())
                .map(|i| {
                    if i / 1024 >= tokens {
                        10.0 as f16
                    } else {
                        (((i * 31 % 269) as f32 - 134.) * 0.0031) as f16
                    }
                })
                .collect();
            let dq = DeviceBuffer::from_host(&stream, &q)?;
            let dk = DeviceBuffer::from_host(
                &stream,
                &k.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            )?;
            let dv = DeviceBuffer::from_host(
                &stream,
                &v.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            )?;
            let guard = 12345.0f32;
            let tensor = DeviceBuffer::from_host(&stream, &vec![guard; q.len() + 17])?;
            let registers = DeviceBuffer::from_host(&stream, &vec![guard; q.len() + 17])?;
            let residual = DeviceBuffer::from_host(&stream, &vec![guard; q.len() + 17])?;
            let scalar = DeviceBuffer::from_host(&stream, &vec![guard; q.len() + 17])?;
            // SAFETY: complete32/8-head cache rows, exact current query extent,
            // causal position and full128-thread tensor CTAs; output tail guarded.
            unsafe {
                module.qwen3_flash.qwen3_flash(
                    &stream,
                    LaunchConfig {
                        grid_dim: ((batch * 4).div_ceil(64) as u32 * 8, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    batch,
                    position,
                    &dq,
                    &dk,
                    &dv,
                    tensor.cu_deviceptr() as *mut f32,
                )?;
                module.qwen3_flash.qwen3_flash_residual(
                    &stream,
                    LaunchConfig {
                        grid_dim: ((batch * 4).div_ceil(64) as u32 * 8, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    batch,
                    position,
                    &dq,
                    &dk,
                    &dv,
                    residual.cu_deviceptr() as *mut f32,
                )?;
                module.qwen3_flash.qwen3_flash_register(
                    &stream,
                    LaunchConfig {
                        grid_dim: ((batch * 4).div_ceil(64) as u32 * 8, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    batch,
                    position,
                    &dq,
                    &dk,
                    &dv,
                    registers.cu_deviceptr() as *mut f32,
                )?;
                module.dense_attention.dense_attention128(
                    &stream,
                    LaunchConfig::for_num_elems((batch * 32 * 32) as u32),
                    batch,
                    32,
                    8,
                    position,
                    1,
                    0,
                    &dq,
                    &dk,
                    &dv,
                    scalar.cu_deviceptr() as *mut f32,
                )?;
            }
            let shared_values = residual.to_host_vec(&stream)?;
            let register_values = registers.to_host_vec(&stream)?;
            ensure!(
                shared_values
                    .iter()
                    .zip(&register_values)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "register/shared residual GQA changed bits at position={position} batch={batch}"
            );
            let b = scalar.to_host_vec(&stream)?;
            for (label, a) in [
                ("original", tensor.to_host_vec(&stream)?),
                ("residual", residual.to_host_vec(&stream)?),
                ("register", registers.to_host_vec(&stream)?),
            ] {
                ensure!(
                    a[q.len()..]
                        .iter()
                        .chain(&b[q.len()..])
                        .all(|x| *x == guard),
                    "GQA output guard overwritten"
                );
                let scalar_error = a[..q.len()]
                    .iter()
                    .zip(&b)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                ensure!(
                    a[..q.len()].iter().all(|x| x.is_finite()) && scalar_error < 3e-5,
                    "tensor/scalar GQA error {scalar_error}"
                );
                let mut f64_error = 0.0f64;
                for row in [0, batch / 2, batch - 1] {
                    for head in [0, 3, 4, 15, 16, 31] {
                        let mut scores: Vec<f64> = (0..position + row + 1)
                            .map(|t| {
                                (0..128)
                                    .map(|d| {
                                        f64::from(q[(row * 32 + head) * 128 + d])
                                            * f64::from(k[(t * 8 + head / 4) * 128 + d] as f32)
                                    })
                                    .sum::<f64>()
                                    / 128.0f64.sqrt()
                            })
                            .collect();
                        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        for score in &mut scores {
                            *score = (*score - max).exp();
                        }
                        let mass: f64 = scores.iter().sum();
                        for d in 0..128 {
                            let expected: f64 = scores
                                .iter()
                                .enumerate()
                                .map(|(t, w)| w * f64::from(v[(t * 8 + head / 4) * 128 + d] as f32))
                                .sum::<f64>()
                                / mass;
                            f64_error = f64_error
                                .max((f64::from(a[(row * 32 + head) * 128 + d]) - expected).abs());
                        }
                    }
                }
                println!(
                    "Qwen3 tensor attention {label} position={position} batch={batch} gain={gain}: scalar_error={scalar_error}, f64_error={f64_error}, guards intact"
                );
                ensure!(f64_error < 3e-5, "independent GQA f64 error {f64_error}");
            }
        }
        Ok(())
    }
}
