//! Shared Float32 convolution and affine operations within the codec encoder.
use super::*;
#[derive(Clone, Copy)]
pub(super) enum Activation {
    Linear,
    Silu,
    Leaky,
    Gelu,
}
impl Engine {
    pub(super) fn codec_linear(
        &self,
        name: &str,
        input: &DeviceBuffer<f32>,
        frames: usize,
        activation: Activation,
    ) -> Result<Buffer> {
        let output = self.component_mm(&format!("codec.{name}.weight"), input, frames, true)?;
        let bias = self.floats(&format!("codec.{name}.bias"))?;
        ensure!(
            !bias.is_empty() && output.len().is_multiple_of(bias.len()),
            "codec affine dimension mismatch"
        );
        let mode = match activation {
            Activation::Linear => 0,
            Activation::Silu => 1,
            Activation::Leaky => 2,
            Activation::Gelu => 3,
        };
        // SAFETY: uniquely owned projection output and complete matching bias rows.
        unsafe {
            self.module.codec.codec_affine(
                &self.stream,
                LaunchConfig::for_num_elems(output.len() as u32),
                output.len(),
                bias.len(),
                mode,
                bias,
                output.cu_deviceptr() as *mut f32,
            )?;
        }
        Ok(output)
    }
    pub(super) fn codec_norm(
        &self,
        name: &str,
        input: &DeviceBuffer<f32>,
        eps: f32,
    ) -> Result<Buffer> {
        ensure!(
            input.len().is_multiple_of(512),
            "codec norm input dimension mismatch"
        );
        let output = self.zeros(input.len())?;
        // SAFETY: full 512-value rows and loaded matching normalization vectors.
        unsafe {
            self.module.vision_math.layer_norm(
                &self.stream,
                LaunchConfig::for_num_elems((input.len() / 512 * 32) as u32),
                512,
                input.len() / 512,
                eps,
                input,
                self.floats(&format!("codec.{name}.weight"))?,
                self.floats(&format!("codec.{name}.bias"))?,
                output.cu_deviceptr() as *mut f32,
            )?;
        }
        Ok(output)
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn codec_conv(
        &self,
        name: &str,
        input: &DeviceBuffer<f32>,
        frames: usize,
        kernel: usize,
        left: usize,
        repeat: usize,
        activation: Activation,
    ) -> Result<Buffer> {
        ensure!(
            input.len() == frames * 512 && frames > 0 && kernel > 0 && repeat > 0,
            "invalid codec convolution input"
        );
        let mut columns = self.zeros(frames * repeat * 512 * kernel)?;
        // SAFETY: complete input rows; kernel explicitly pads out-of-range temporal positions.
        unsafe {
            self.module.codec.codec_columns(
                &self.stream,
                LaunchConfig::for_num_elems(columns.len() as u32),
                frames,
                kernel,
                left,
                repeat,
                input,
                &mut columns,
            )?;
        }
        self.codec_linear(name, &columns, frames * repeat, activation)
    }
}
