//! MiniCPM patch extraction, learned positions, spatial grouping, and exact-GELU approximation.
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub fn minicpm_patches(
        width: usize,
        height: usize,
        pixels: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let patch = i / 588;
            let component = i % 588;
            let x = patch % (width / 14) * 14 + component % 14;
            let y = patch / (width / 14) * 14 + component % 196 / 14;
            *value = pixels[component / 196 * width * height + y * width + x];
        }
    }

    /// Preserve the part of an FP32 activation lost by a single FP16 operand.
    #[kernel]
    pub fn minicpm_f16_residual(input: &[f32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let packed = cuda_device::convert::cvt_f16x2_f32(input[i], 0.0);
            let (high, _) = cuda_device::convert::cvt_f32x2_f16x2(packed);
            *value = input[i] - high;
        }
    }

    #[kernel]
    pub fn minicpm_position<const BF16: bool>(
        width: usize,
        height: usize,
        input: &[f32],
        bias: &[f32],
        table: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let patch = i / 1152;
            let channel = i % 1152;
            let y = patch / width * 70 / height;
            let x = patch % width * 70 / width;
            let value_with_bias = input[i] + bias[channel];
            let position = table[(y * 70 + x) * 1152 + channel];
            *value = if BF16 {
                super::super::minicpm_bf16::rounded(
                    super::super::minicpm_bf16::rounded(value_with_bias) + position,
                )
            } else {
                value_with_bias + position
            };
        }
    }

    /// Place each 2x2 raster block into four adjacent rows, preserving channels.
    #[kernel]
    pub fn minicpm_merge_order(width: usize, input: &[f32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let patch = i / 1152;
            let group = patch / 4;
            let x = group % (width / 2) * 2 + patch % 2;
            let y = group / (width / 2) * 2 + patch % 4 / 2;
            *value = input[(y * width + x) * 1152 + i % 1152];
        }
    }

    #[kernel]
    pub fn minicpm_window_mean(input: &[f32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let at = i / 1152 * 4608 + i % 1152;
            *value = (input[at] + input[at + 1152] + input[at + 2304] + input[at + 3456]) * 0.25;
        }
    }

    /// Abramowitz-Stegun 7.1.26 bounds erf error by 1.5e-7 before FP32 rounding.
    #[kernel]
    pub unsafe fn minicpm_gelu_erf<const BF16: bool>(
        count: usize,
        dim: usize,
        bias: &[f32],
        values: *mut f32,
    ) {
        let i = thread::index_1d().get();
        if i < count {
            let x = unsafe { *values.add(i) } + bias[i % dim];
            let x = if BF16 {
                super::super::minicpm_bf16::rounded(x)
            } else {
                x
            };
            let z = x * core::f32::consts::FRAC_1_SQRT_2;
            let t = 1.0 / (1.0 + 0.3275911 * z.abs());
            let p = (((((1.0614054 * t - 1.4531521) * t) + 1.4214138) * t - 0.28449672) * t
                + 0.2548296)
                * t;
            let erf = 1.0 - p * (-z * z).exp();
            let erf = if z < 0.0 { -erf } else { erf };
            unsafe {
                let value = 0.5 * x * (1.0 + erf);
                *values.add(i) = if BF16 {
                    super::super::minicpm_bf16::rounded(value)
                } else {
                    value
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};
    #[test]
    fn patch_coordinates_positions_grouping_and_gelu() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = super::super::load(&context)?;
        let (width, height) = (112usize, 56usize);
        let pixels: Vec<f32> = (0..width * height * 3).map(|i| i as f32).collect();
        let input = DeviceBuffer::from_host(&stream, &pixels)?;
        let mut output = DeviceBuffer::zeroed(&stream, pixels.len())?;
        // SAFETY: complete RGB 14x14 patches and disjoint exact-size output.
        unsafe {
            module.minicpm_vision.minicpm_patches(
                &stream,
                LaunchConfig::for_num_elems(pixels.len() as u32),
                width,
                height,
                &input,
                &mut output,
            )?;
        }
        let actual = output.to_host_vec(&stream)?;
        for (i, &value) in actual.iter().enumerate() {
            let patch = i / 588;
            let c = i % 588;
            let x = patch % (width / 14) * 14 + c % 14;
            let y = patch / (width / 14) * 14 + c % 196 / 14;
            assert_eq!(value, pixels[c / 196 * width * height + y * width + x]);
        }
        let rows = 32;
        let source: Vec<f32> = (0..rows * 1152).map(|i| i as f32).collect();
        let input = DeviceBuffer::from_host(&stream, &source)?;
        let mut grouped = DeviceBuffer::zeroed(&stream, source.len())?;
        let mut means = DeviceBuffer::zeroed(&stream, rows / 4 * 1152)?;
        // SAFETY: raster width8 and height4 form complete2x2 windows.
        unsafe {
            module.minicpm_vision.minicpm_merge_order(
                &stream,
                LaunchConfig::for_num_elems(source.len() as u32),
                8,
                &input,
                &mut grouped,
            )?;
            module.minicpm_vision.minicpm_window_mean(
                &stream,
                LaunchConfig::for_num_elems(means.len() as u32),
                &grouped,
                &mut means,
            )?;
        }
        let actual = grouped.to_host_vec(&stream)?;
        let mean = means.to_host_vec(&stream)?;
        for group in 0..8 {
            let corners = [
                (group / 4 * 2) * 8 + group % 4 * 2,
                (group / 4 * 2) * 8 + group % 4 * 2 + 1,
                (group / 4 * 2 + 1) * 8 + group % 4 * 2,
                (group / 4 * 2 + 1) * 8 + group % 4 * 2 + 1,
            ];
            for channel in 0..1152 {
                for (j, &row) in corners.iter().enumerate() {
                    assert_eq!(
                        actual[(group * 4 + j) * 1152 + channel],
                        source[row * 1152 + channel]
                    );
                }
                let expected = corners
                    .iter()
                    .map(|&row| source[row * 1152 + channel] as f64)
                    .sum::<f64>()
                    * 0.25;
                assert_eq!(mean[group * 1152 + channel], expected as f32);
            }
        }
        let table: Vec<f32> = (0..4900 * 1152).map(|i| (i / 1152) as f32).collect();
        let table = DeviceBuffer::from_host(&stream, &table)?;
        let bias = DeviceBuffer::from_host(&stream, &vec![0.5f32; 1152])?;
        for (width, height) in [(8usize, 4usize), (36, 28), (70, 70)] {
            let source: Vec<f32> = (0..width * height * 1152)
                .map(|i| (i % 1152) as f32)
                .collect();
            let input = DeviceBuffer::from_host(&stream, &source)?;
            let mut output = DeviceBuffer::zeroed(&stream, source.len())?;
            // SAFETY: complete1152-channel rows and a validated70x70 position table.
            unsafe {
                module.minicpm_vision.minicpm_position::<false>(
                    &stream,
                    LaunchConfig::for_num_elems(source.len() as u32),
                    width,
                    height,
                    &input,
                    &bias,
                    &table,
                    &mut output,
                )?;
            }
            for (i, &value) in output.to_host_vec(&stream)?.iter().enumerate() {
                let patch = i / 1152;
                let bucket = |index: usize, size: usize| {
                    (1..70)
                        .filter(|&k| index as f32 / size as f32 >= (k as f64 / 70.0) as f32)
                        .count()
                };
                let expected = source[i]
                    + 0.5
                    + (bucket(patch / width, height) * 70 + bucket(patch % width, width)) as f32;
                assert_eq!(value, expected, "position {width}x{height}, element {i}");
            }
        }
        let values = [-6.0f32, -3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0, 6.0];
        let gold = [
            -0.000000005919526,
            -0.004049694,
            -0.15865526,
            -0.046017215,
            0.0,
            0.053982783,
            0.8413448,
            2.9959502,
            6.0,
        ];
        let input = DeviceBuffer::from_host(&stream, &values)?;
        let bias = DeviceBuffer::from_host(&stream, &[0.0f32])?;
        // SAFETY: each thread owns one bounded input and the bias has one channel.
        unsafe {
            module.minicpm_vision.minicpm_gelu_erf::<false>(
                &stream,
                LaunchConfig::for_num_elems(values.len() as u32),
                values.len(),
                1,
                &bias,
                input.cu_deviceptr() as *mut f32,
            )?;
        }
        for (actual, expected) in input.to_host_vec(&stream)?.iter().zip(gold) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
        Ok(())
    }
}
