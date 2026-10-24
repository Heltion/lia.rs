//! S3 Float32 dense projections, normalization and convolution gathering.
use super::*;
impl Engine {
    pub(super) fn s3_norm(&self, name: &str, input: &DeviceBuffer<f32>) -> Result<Buffer> {
        ensure!(
            input.len().is_multiple_of(1280),
            "invalid S3 normalization extent"
        );
        let output = self.zeros(input.len())?;
        // SAFETY: complete 1280-channel rows and architecture-checked affine vectors.
        unsafe {
            self.module.vision_math.layer_norm(
                &self.stream,
                LaunchConfig::for_num_elems((input.len() / 1280 * 32) as u32),
                1280,
                input.len() / 1280,
                1e-5,
                input,
                self.floats(&format!("s3.{name}.weight"))?,
                self.floats(&format!("s3.{name}.bias"))?,
                output.cu_deviceptr() as *mut f32,
            )?;
        }
        Ok(output)
    }
    pub(super) fn s3_linear(
        &self,
        name: &str,
        input: &DeviceBuffer<f32>,
        frames: usize,
        bias: bool,
        gelu: bool,
    ) -> Result<Buffer> {
        let output = self.component_mm(&format!("s3.{name}.weight"), input, frames, true)?;
        if bias {
            let values = self.floats(&format!("s3.{name}.bias"))?;
            ensure!(
                !values.is_empty() && output.len().is_multiple_of(values.len()),
                "invalid S3 affine extent"
            );
            let config = LaunchConfig::for_num_elems(output.len() as u32);
            // SAFETY: uniquely owned complete output rows and matching bias vector.
            unsafe {
                if gelu {
                    self.module.minicpm_vision.minicpm_gelu_erf::<false>(
                        &self.stream,
                        config,
                        output.len(),
                        values.len(),
                        values,
                        output.cu_deviceptr() as *mut f32,
                    )?;
                } else {
                    self.module.vision_math.affine_in_place::<false>(
                        &self.stream,
                        config,
                        output.len(),
                        values.len(),
                        values,
                        output.cu_deviceptr() as *mut f32,
                    )?;
                }
            }
        }
        Ok(output)
    }
    pub(super) fn s3_conv<const PLANAR: bool>(
        &self,
        input: &DeviceBuffer<f32>,
        frames: usize,
        channels: usize,
        name: &str,
    ) -> Result<Buffer> {
        let rows = frames.div_ceil(2);
        let mut columns = self.zeros(rows * channels * 3)?;
        // SAFETY: input has complete planar or time-major rows and explicit symmetric zero padding.
        unsafe {
            self.module.whisper.whisper_im2col::<PLANAR>(
                &self.stream,
                LaunchConfig::for_num_elems(columns.len() as u32),
                frames,
                channels,
                2,
                input,
                &mut columns,
            )?;
        }
        self.s3_linear(name, &columns, rows, true, true)
    }
}
