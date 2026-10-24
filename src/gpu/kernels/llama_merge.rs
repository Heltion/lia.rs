//! Small split-K reductions keep each head entirely within one warp.
use cuda_device::{kernel, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[kernel]
    pub unsafe fn llama_merge_small(
        query_start: usize,
        queries: usize,
        parts: usize,
        partial: &[f32],
        output: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        if group < queries * 16 {
            let offset = (group * parts + lane) * 130;
            let maximum = if lane < parts {
                partial[offset + 128]
            } else {
                f32::NEG_INFINITY
            };
            let mass = if lane < parts {
                partial[offset + 129]
            } else {
                0.0
            };
            let mut global_maximum = maximum;
            let mut step = 16;
            while step > 0 {
                global_maximum = global_maximum.max(warp::shuffle_down_f32_sync(
                    0xffff_ffff,
                    global_maximum,
                    step,
                ));
                step /= 2;
            }
            global_maximum = warp::shuffle_f32_sync(0xffff_ffff, global_maximum, 0);
            let weight = if mass > 0.0 {
                cuda_device::float::ex2_approx_f32(
                    (maximum - global_maximum) * core::f32::consts::LOG2_E,
                )
            } else {
                0.0
            };
            let mut denominator = weight * mass;
            step = 16;
            while step > 0 {
                denominator += warp::shuffle_down_f32_sync(0xffff_ffff, denominator, step);
                step /= 2;
            }
            denominator = warp::shuffle_f32_sync(0xffff_ffff, denominator, 0);
            let mut sums = [0_f32; 4];
            let mut p = 0;
            while p < parts {
                let factor = warp::shuffle_f32_sync(0xffff_ffff, weight, p as u32);
                let at = (group * parts + p) * 130 + lane;
                sums[0] = factor.mul_add(partial[at], sums[0]);
                sums[1] = factor.mul_add(partial[at + 32], sums[1]);
                sums[2] = factor.mul_add(partial[at + 64], sums[2]);
                sums[3] = factor.mul_add(partial[at + 96], sums[3]);
                p += 1;
            }
            let at = query_start * 2048 + group * 128 + lane;
            unsafe {
                *output.add(at) = sums[0] / denominator;
                *output.add(at + 32) = sums[1] / denominator;
                *output.add(at + 64) = sums[2] / denominator;
                *output.add(at + 96) = sums[3] / denominator;
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
    fn warp_reduction_matches_block_reduction_bits() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        for parts in [1, 2, 3, 4, 8, 16] {
            let mut partial = vec![0_f32; 3 * 16 * parts * 130];
            for group in 0..3 * 16 {
                for p in 0..parts {
                    let at = (group * parts + p) * 130;
                    for column in 0..128 {
                        partial[at + column] =
                            ((group * 7 + p * 13 + column * 17) % 59) as f32 / 32.0 - 1.0;
                    }
                    let valid = p == 0 || (group + p) % 3 != 0;
                    partial[at + 128] = if valid {
                        (group % 7 + p) as f32 / 8.0
                    } else {
                        f32::NEG_INFINITY
                    };
                    partial[at + 129] = if valid { 0.25 + p as f32 / 16.0 } else { 0.0 };
                }
            }
            let partial = DeviceBuffer::from_host(&stream, &partial)?;
            // Extra rows on both sides detect writes beyond the requested span.
            let expected = DeviceBuffer::<f32>::zeroed(&stream, 9 * 2048)?;
            let actual = DeviceBuffer::<f32>::zeroed(&stream, expected.len())?;
            // SAFETY: full partial heads, complete warps, <=16 partitions and
            // output capacity for the three selected rows starting at row five.
            unsafe {
                module.llama.llama_attention_merge(
                    &stream,
                    LaunchConfig {
                        grid_dim: (48, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    5,
                    3,
                    parts,
                    &partial,
                    expected.cu_deviceptr() as *mut f32,
                )?;
                module.llama_merge.llama_merge_small(
                    &stream,
                    LaunchConfig {
                        grid_dim: (12, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    5,
                    3,
                    parts,
                    &partial,
                    actual.cu_deviceptr() as *mut f32,
                )?;
            }
            let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
            assert_eq!(
                bits(actual.to_host_vec(&stream)?),
                bits(expected.to_host_vec(&stream)?),
                "parts={parts}"
            );
        }
        Ok(())
    }
}
