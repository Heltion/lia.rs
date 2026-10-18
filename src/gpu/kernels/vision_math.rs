//! Vision LayerNorm, affine activations, and spatial rotary embeddings.
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[inline(always)]
    fn warp_sum(mut value: f32) -> f32 {
        let mut step = 16;
        while step > 0 {
            value += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, value, step);
            step /= 2;
        }
        cuda_device::warp::shuffle_f32_sync(0xffff_ffff, value, 0)
    }

    /// One full warp per row; unlike RMSNorm, subtract the mean first.
    #[kernel]
    pub unsafe fn layer_norm(
        dim: usize,
        rows: usize,
        eps: f32,
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        output: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let row = tid / 32;
        let lane = tid % 32;
        if row < rows {
            let mut sum = 0.0;
            let mut j = lane;
            while j < dim {
                sum += input[row * dim + j];
                j += 32;
            }
            let mean = warp_sum(sum) / dim as f32;
            let mut variance = 0.0;
            j = lane;
            while j < dim {
                let centered = input[row * dim + j] - mean;
                variance += centered * centered;
                j += 32;
            }
            let scale = 1.0 / (warp_sum(variance) / dim as f32 + eps).sqrt();
            j = lane;
            while j < dim {
                unsafe {
                    *output.add(row * dim + j) =
                        ((input[row * dim + j] - mean) * scale) * weight[j] + bias[j];
                }
                j += 32;
            }
        }
    }

    #[kernel]
    pub unsafe fn affine_in_place<const GELU: bool>(
        count: usize,
        dim: usize,
        bias: &[f32],
        values: *mut f32,
    ) {
        let i = thread::index_1d().get();
        if i < count {
            let x = unsafe { *values.add(i) } + bias[i % dim];
            unsafe {
                *values.add(i) = if GELU {
                    0.5 * x * (1.0 + (0.797_884_6 * x * (1.0 + 0.044_715 * x * x)).tanh())
                } else {
                    x
                };
            }
        }
    }

    /// Q/K use split-half pairs of 36 elements, with 18 row and 18 column
    /// frequencies. V is copied unrotated. Input and bias are Q, K, V per patch.
    #[kernel]
    pub fn qkv_rope(
        patch_width: usize,
        input: &[f32],
        bias: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let token = i / 3456;
            let component = i % 3456;
            let x = input[i] + bias[component];
            if component >= 2304 {
                *value = x;
                return;
            }
            let dim = component % 72;
            let pair = if dim < 36 { i + 36 } else { i - 36 };
            let y = input[pair] + bias[pair % 3456];
            let group = token / 4;
            let px = (group % (patch_width / 2)) * 2 + token % 2;
            let py = (group / (patch_width / 2)) * 2 + (token % 4) / 2;
            let frequency = dim % 36;
            let position = if frequency < 18 { py } else { px };
            let angle = position as f32 * 10000.0f32.powf(-((frequency % 18) as f32) / 18.0);
            *value = x * angle.cos()
                + if dim < 36 {
                    -y * angle.sin()
                } else {
                    y * angle.sin()
                };
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};

    fn close(actual: f32, expected: f64, at: usize) {
        assert!(
            (actual as f64 - expected).abs() < 3e-5 * (1.0 + expected.abs()),
            "at {at}: {actual} != {expected}"
        );
    }

    #[test]
    fn centered_norm_affine_and_spatial_rope() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = super::super::load(&context)?;
        let (dim, rows) = (1152, 5);
        let mut input: Vec<f32> = (0..dim * rows)
            .map(|i| (i % 113) as f32 / 17.0 + 13.0)
            .collect();
        input[..dim].fill(7.0);
        let weight: Vec<f32> = (0..dim).map(|i| 0.5 + (i % 7) as f32 / 11.0).collect();
        let bias: Vec<f32> = (0..dim).map(|i| (i % 13) as f32 / 19.0 - 0.3).collect();
        let din = DeviceBuffer::from_host(&stream, &input)?;
        let dw = DeviceBuffer::from_host(&stream, &weight)?;
        let db = DeviceBuffer::from_host(&stream, &bias)?;
        let output = DeviceBuffer::<f32>::zeroed(&stream, input.len())?;
        // SAFETY: launch uses complete warps, one per valid row; all row and
        // affine extents match, and output is distinct from every input.
        unsafe {
            module.vision_math.layer_norm(
                &stream,
                LaunchConfig::for_num_elems((rows * 32) as u32),
                dim,
                rows,
                1e-6,
                &din,
                &dw,
                &db,
                output.cu_deviceptr() as *mut f32,
            )?;
        }
        for (row, actual) in output.to_host_vec(&stream)?.chunks_exact(dim).enumerate() {
            let source = &input[row * dim..(row + 1) * dim];
            let mean = source.iter().map(|v| *v as f64).sum::<f64>() / dim as f64;
            let variance = source
                .iter()
                .map(|v| (*v as f64 - mean).powi(2))
                .sum::<f64>()
                / dim as f64;
            for j in 0..dim {
                close(
                    actual[j],
                    (source[j] as f64 - mean) / (variance + 1e-6).sqrt() * weight[j] as f64
                        + bias[j] as f64,
                    row * dim + j,
                );
            }
        }
        let input: Vec<f32> = (0..dim * rows)
            .map(|i| (i % 137) as f32 / 9.0 - 7.5)
            .collect();
        for gelu in [false, true] {
            let output = DeviceBuffer::from_host(&stream, &input)?;
            // SAFETY: each guarded element reads one matching input and a bias
            // within dim; both const specializations use identical extents.
            unsafe {
                let launch = LaunchConfig::for_num_elems(input.len() as u32);
                if gelu {
                    module.vision_math.affine_in_place::<true>(
                        &stream,
                        launch,
                        input.len(),
                        dim,
                        &db,
                        output.cu_deviceptr() as *mut f32,
                    )?;
                } else {
                    module.vision_math.affine_in_place::<false>(
                        &stream,
                        launch,
                        input.len(),
                        dim,
                        &db,
                        output.cu_deviceptr() as *mut f32,
                    )?;
                }
            }
            for (i, actual) in output.to_host_vec(&stream)?.into_iter().enumerate() {
                let x = (input[i] + bias[i % dim]) as f64;
                let expected = if gelu {
                    0.5 * x
                        * (1.0
                            + (std::f64::consts::FRAC_2_SQRT_PI / 2.0f64.sqrt()
                                * x
                                * (1.0 + 0.044715 * x * x))
                                .tanh())
                } else {
                    x
                };
                close(actual, expected, i);
            }
        }
        let width = 6;
        let tokens = width * 4;
        let input: Vec<f32> = (0..tokens * 3456)
            .map(|i| ((i * 17 % 103) as f32 - 51.0) / 23.0)
            .collect();
        let bias: Vec<f32> = (0..3456).map(|i| (i % 11) as f32 / 29.0).collect();
        let din = DeviceBuffer::from_host(&stream, &input)?;
        let db = DeviceBuffer::from_host(&stream, &bias)?;
        let mut output = DeviceBuffer::<f32>::zeroed(&stream, input.len())?;
        // SAFETY: width is even, tokens form full merge groups, and QKV/bias
        // extents are exactly tokens*3456 and 3456 respectively.
        unsafe {
            module.vision_math.qkv_rope(
                &stream,
                LaunchConfig::for_num_elems(input.len() as u32),
                width,
                &din,
                &db,
                &mut output,
            )?;
        }
        let grid = crate::vision::ImageGrid::new(width * 16, 64)?;
        for (i, actual) in output.to_host_vec(&stream)?.into_iter().enumerate() {
            let component = i % 3456;
            let x = (input[i] + bias[component]) as f64;
            let expected = if component >= 2304 {
                x
            } else {
                let (px, py) = grid.patch_coordinates(i / 3456)?;
                let d = component % 72;
                let f = d % 36;
                let position = if f < 18 { py } else { px };
                let angle = position as f64 * 10000.0f64.powf(-((f % 18) as f64) / 18.0);
                let pair = if d < 36 { i + 36 } else { i - 36 };
                let y = (input[pair] + bias[pair % 3456]) as f64;
                x * angle.cos()
                    + if d < 36 {
                        -y * angle.sin()
                    } else {
                        y * angle.sin()
                    }
            };
            close(actual, expected, i);
        }
        Ok(())
    }
}
