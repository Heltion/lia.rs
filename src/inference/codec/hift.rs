//! HiFT acoustic components, starting with its convolutional F0 predictor.
use super::*;
mod decoder;
mod decoder_weights;
mod source;
mod spectral;
mod weights;
impl Engine {
    /// Predict nonnegative F0 values from time-major 80-channel mel frames.
    pub fn codec_f0(&self, mel: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            self.weights
                .contains_key("hift.f0_predictor.classifier.weight"),
            "HiFT F0 weights are not loaded"
        );
        ensure!(
            !mel.is_empty()
                && mel.len().is_multiple_of(80)
                && mel.len() <= crate::models::minicpm_o45::MAX_MEL_FRAMES * 80
                && mel.iter().all(|v| v.is_finite()),
            "invalid HiFT mel input"
        );
        let frames = mel.len() / 80;
        let mut hidden = self.zeros(mel.len())?;
        hidden.copy_from_host(&self.stream, mel)?;
        self.hift_f0(&hidden, frames)?
            .to_host_vec(&self.stream)
            .map_err(Into::into)
    }
    pub(super) fn hift_f0(&self, input: &DeviceBuffer<f32>, frames: usize) -> Result<Buffer> {
        let mut hidden = self.zeros(input.len())?;
        hidden.copy_from_device_async(input, &self.stream)?;
        for layer in 0..5 {
            hidden = self.hift_conv(
                &format!("f0_predictor.condnet.{}", layer * 2),
                &hidden,
                frames,
                if layer == 0 { 80 } else { 512 },
                3,
                1,
                1,
                1,
            )?;
        }
        self.hift_linear("f0_predictor.classifier", &hidden, frames, 2)
    }
    fn hift_linear(
        &self,
        name: &str,
        input: &DeviceBuffer<f32>,
        rows: usize,
        activation: usize,
    ) -> Result<Buffer> {
        let weight_name = format!("hift.{name}.weight");
        let weight = self.weight(&weight_name)?;
        let columns = input.len() / rows;
        ensure!(
            input.len().is_multiple_of(rows) && columns <= weight.shape[0],
            "invalid HiFT matrix dimensions"
        );
        let output = self.zeros(rows * weight.shape[1])?;
        // SAFETY: full logical input rows, padded Float32 weight rows and a disjoint output.
        unsafe {
            self.module.dense_simt.dense_f32(
                &self.stream,
                LaunchConfig::for_num_elems((output.len() * 32) as u32),
                columns,
                weight.shape[0],
                weight.shape[1],
                input,
                &weight.data,
                output.cu_deviceptr() as *mut f32,
            )?;
        }
        let bias = self.floats(&format!("hift.{name}.bias"))?;
        ensure!(
            !bias.is_empty() && output.len() == rows * bias.len(),
            "invalid HiFT affine dimensions"
        );
        // SAFETY: matching complete bias rows and a uniquely owned output.
        unsafe {
            self.module.hift.hift_affine(
                &self.stream,
                LaunchConfig::for_num_elems(output.len() as u32),
                bias.len(),
                activation,
                output.len(),
                bias,
                output.cu_deviceptr() as *mut f32,
            )?;
        }
        Ok(output)
    }
    #[allow(clippy::too_many_arguments)]
    fn hift_conv(
        &self,
        name: &str,
        input: &DeviceBuffer<f32>,
        frames: usize,
        channels: usize,
        kernel: usize,
        left: usize,
        dilation: usize,
        activation: usize,
    ) -> Result<Buffer> {
        ensure!(
            input.len() == frames * channels
                && frames > 0
                && channels > 0
                && kernel > 0
                && dilation > 0,
            "invalid HiFT convolution dimensions"
        );
        let mut columns = self.zeros(frames * channels * kernel)?;
        // SAFETY: checked complete input/output rows, with explicit temporal zero padding.
        unsafe {
            self.module.hift.hift_columns(
                &self.stream,
                LaunchConfig::for_num_elems(columns.len() as u32),
                channels,
                kernel,
                left,
                dilation,
                1,
                input,
                &mut columns,
            )?;
        }
        self.hift_linear(name, &columns, frames, activation)
    }
}
