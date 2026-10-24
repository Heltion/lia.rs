//! Three-stage HiFT mel/source decoder with isolated temporal boundaries.
use super::*;
impl Engine {
    /// Return raw time-major spectral channels for subsequent overlap-add synthesis.
    pub fn codec_hift_spectrum(&self, mel: &[f32], excitation: &[f32]) -> Result<Vec<f32>> {
        let (mel, excitation, frames) = self.hift_inputs(mel, excitation)?;
        self.hift_decode(&mel, &excitation, frames)?
            .to_host_vec(&self.stream)
            .map_err(Into::into)
    }
    /// Synthesize 24 kHz mono samples from mel frames and an explicit excitation source.
    pub fn codec_hift_waveform(&self, mel: &[f32], excitation: &[f32]) -> Result<Vec<f32>> {
        let (mel, excitation, frames) = self.hift_inputs(mel, excitation)?;
        let spectrum = self.hift_decode(&mel, &excitation, frames)?;
        self.hift_inverse(&spectrum)?
            .to_host_vec(&self.stream)
            .map_err(Into::into)
    }
    fn hift_inputs(&self, mel: &[f32], excitation: &[f32]) -> Result<(Buffer, Buffer, usize)> {
        ensure!(
            self.weights.contains_key("hift.conv_post.weight"),
            "HiFT decoder weights are not loaded"
        );
        ensure!(
            !mel.is_empty()
                && mel.len().is_multiple_of(80)
                && mel.len() <= crate::models::minicpm_o45::MAX_MEL_FRAMES * 80
                && excitation.len() == mel.len() / 80 * 480
                && mel.iter().chain(excitation).all(|v| v.is_finite()),
            "invalid HiFT mel or excitation input"
        );
        let frames = mel.len() / 80;
        let mut x = self.zeros(mel.len())?;
        let mut s = self.zeros(excitation.len())?;
        x.copy_from_host(&self.stream, mel)?;
        s.copy_from_host(&self.stream, excitation)?;
        Ok((x, s, frames))
    }
    pub(in crate::inference::codec) fn hift_decode(
        &self,
        mel: &DeviceBuffer<f32>,
        excitation: &DeviceBuffer<f32>,
        mel_frames: usize,
    ) -> Result<Buffer> {
        let spectrum = self.hift_forward_spectrum(excitation)?;
        let mut x = self.hift_conv("conv_pre", mel, mel_frames, 80, 7, 3, 1, 0)?;
        let mut frames = mel_frames;
        for stage in 0..3 {
            let channels = 256 >> stage;
            let activated = self.hift_leaky(&x, 0.1)?;
            x = self.hift_upsample(
                &format!("ups.{stage}"),
                &activated,
                frames,
                channels * 2,
                [16, 11, 7][stage],
                [8, 5, 3][stage],
                [4, 3, 2][stage],
            )?;
            frames *= [8, 5, 3][stage];
            if stage == 2 {
                let mut padded = self.zeros((frames + 1) * channels)?;
                // SAFETY: reflection of one frame requires at least two input frames.
                unsafe {
                    self.module.hift_decode.hift_reflect_left(
                        &self.stream,
                        LaunchConfig::for_num_elems(padded.len() as u32),
                        channels,
                        &x,
                        &mut padded,
                    )?;
                }
                x = padded;
                frames += 1;
            }
            let source = self.hift_strided_conv(
                &format!("source_downs.{stage}"),
                &spectrum,
                frames,
                18,
                [30, 6, 1][stage],
                [7, 1, 0][stage],
                [15, 3, 1][stage],
            )?;
            let source = self.hift_residual(
                &format!("source_resblocks.{stage}"),
                &source,
                frames,
                channels,
                [7, 7, 11][stage],
            )?;
            let fused = self.add(&x, &source)?;
            let a = self.hift_residual(
                &format!("resblocks.{}", stage * 3),
                &fused,
                frames,
                channels,
                3,
            )?;
            let b = self.hift_residual(
                &format!("resblocks.{}", stage * 3 + 1),
                &fused,
                frames,
                channels,
                7,
            )?;
            let c = self.hift_residual(
                &format!("resblocks.{}", stage * 3 + 2),
                &fused,
                frames,
                channels,
                11,
            )?;
            let mut mean = self.zeros(fused.len())?;
            // SAFETY: three identically shaped residual outputs and a distinct destination.
            unsafe {
                self.module.hift_decode.hift_mean(
                    &self.stream,
                    LaunchConfig::for_num_elems(mean.len() as u32),
                    &a,
                    &b,
                    &c,
                    &mut mean,
                )?;
            }
            x = mean;
        }
        let x = self.hift_leaky(&x, 0.01)?;
        self.hift_conv("conv_post", &x, frames, 64, 7, 3, 1, 0)
    }
    fn hift_residual(
        &self,
        name: &str,
        input: &DeviceBuffer<f32>,
        frames: usize,
        channels: usize,
        kernel: usize,
    ) -> Result<Buffer> {
        let mut hidden = self.zeros(input.len())?;
        hidden.copy_from_device_async(input, &self.stream)?;
        for (layer, dilation) in [1, 3, 5].into_iter().enumerate() {
            let activated =
                self.hift_snake(&format!("{name}.activations1.{layer}.alpha"), &hidden)?;
            let conv = self.hift_conv(
                &format!("{name}.convs1.{layer}"),
                &activated,
                frames,
                channels,
                kernel,
                (kernel - 1) * dilation / 2,
                dilation,
                0,
            )?;
            let activated =
                self.hift_snake(&format!("{name}.activations2.{layer}.alpha"), &conv)?;
            let conv = self.hift_conv(
                &format!("{name}.convs2.{layer}"),
                &activated,
                frames,
                channels,
                kernel,
                (kernel - 1) / 2,
                1,
                0,
            )?;
            hidden = self.add(&hidden, &conv)?;
        }
        Ok(hidden)
    }
    fn hift_snake(&self, name: &str, input: &DeviceBuffer<f32>) -> Result<Buffer> {
        let alpha = self.floats(&format!("hift.{name}"))?;
        ensure!(
            !alpha.is_empty() && input.len().is_multiple_of(alpha.len()),
            "invalid Snake channel extent"
        );
        let mut out = self.zeros(input.len())?;
        // SAFETY: complete channel rows and matching loaded alpha vector.
        unsafe {
            self.module.hift_decode.hift_snake(
                &self.stream,
                LaunchConfig::for_num_elems(out.len() as u32),
                input,
                alpha,
                &mut out,
            )?;
        }
        Ok(out)
    }
    fn hift_leaky(&self, input: &DeviceBuffer<f32>, slope: f32) -> Result<Buffer> {
        let mut out = self.zeros(input.len())?;
        // SAFETY: disjoint destination matches the entire input slice.
        unsafe {
            self.module.hift_decode.hift_leaky(
                &self.stream,
                LaunchConfig::for_num_elems(out.len() as u32),
                slope,
                input,
                &mut out,
            )?;
        }
        Ok(out)
    }
    #[allow(clippy::too_many_arguments)]
    fn hift_upsample(
        &self,
        name: &str,
        input: &DeviceBuffer<f32>,
        frames: usize,
        channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Buffer> {
        ensure!(
            input.len() == frames * channels && frames > 0 && stride > 0,
            "invalid HiFT transpose convolution"
        );
        let output_frames = (frames - 1) * stride + kernel - 2 * padding;
        let mut columns = self.zeros(output_frames * channels * kernel)?;
        // SAFETY: complete logical rows; the gather skips positions not on the upsample lattice.
        unsafe {
            self.module.hift_decode.hift_transpose_columns(
                &self.stream,
                LaunchConfig::for_num_elems(columns.len() as u32),
                channels,
                kernel,
                padding,
                stride,
                input,
                &mut columns,
            )?;
        }
        self.hift_linear(name, &columns, output_frames, 0)
    }
    #[allow(clippy::too_many_arguments)]
    fn hift_strided_conv(
        &self,
        name: &str,
        input: &DeviceBuffer<f32>,
        frames: usize,
        channels: usize,
        kernel: usize,
        left: usize,
        stride: usize,
    ) -> Result<Buffer> {
        ensure!(
            input.len().is_multiple_of(channels) && frames > 0 && stride > 0,
            "invalid HiFT strided convolution"
        );
        let mut columns = self.zeros(frames * channels * kernel)?;
        // SAFETY: zero-padded temporal gather over complete input and output channel rows.
        unsafe {
            self.module.hift.hift_columns(
                &self.stream,
                LaunchConfig::for_num_elems(columns.len() as u32),
                channels,
                kernel,
                left,
                1,
                stride,
                input,
                &mut columns,
            )?;
        }
        self.hift_linear(name, &columns, frames, 0)
    }
}
