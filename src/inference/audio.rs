//! MiniCPM-o Whisper encoder, projection and five-frame audio pooling.
use super::*;
use crate::models::minicpm_o45::audio::Features;
use anyhow::ensure;
mod stream;
pub use stream::AudioCache;

enum Activation {
    Linear,
    Gelu,
    Relu,
}

impl Engine {
    pub fn audio_input_enabled(&self) -> bool {
        self.geometry.qwen3 && self.weights.contains_key("audio.conv1.weight")
    }

    /// Export audio features for validation; serving can retain the device result.
    pub fn audio_embeddings(&self, features: &Features, chunk_size: usize) -> Result<Vec<f32>> {
        self.encode_audio(features, chunk_size)?
            .to_host_vec(&self.stream)
            .map_err(Into::into)
    }

    pub(super) fn encode_audio(&self, features: &Features, chunk_size: usize) -> Result<Buffer> {
        ensure!(
            self.geometry.qwen3 && self.weights.contains_key("audio.conv1.weight"),
            "MiniCPM-o audio weights are not loaded"
        );
        ensure!(
            (9..=3000).contains(&features.frames)
                && features.values.len() == 80 * features.frames
                && features.values.iter().all(|v| v.is_finite())
                && chunk_size <= 1500,
            "invalid prepared audio features"
        );
        let input = DeviceBuffer::from_host(&self.stream, &features.values)?;
        let first = self.audio_conv::<true>(&input, features.frames, 80, 1, "conv1")?;
        let frames = features.frames.div_ceil(2);
        let second = self.audio_conv::<false>(&first, features.frames, 1024, 2, "conv2")?;
        drop(first);
        let mut hidden = self.zeros(second.len())?;
        // SAFETY: complete 1024-channel rows within the learned 1500-position table.
        unsafe {
            self.module.whisper.whisper_position(
                &self.stream,
                LaunchConfig::for_num_elems(hidden.len() as u32),
                &second,
                self.floats("audio.position")?,
                &mut hidden,
            )?;
        }
        drop(second);
        for layer in 0..24 {
            hidden = self.audio_block(layer, &hidden, frames, chunk_size)?;
        }
        let encoded = self.audio_norm("post_ln", &hidden)?;
        let projected = self.component_mm("audio.proj1.weight", &encoded, frames, true)?;
        let projected = self.audio_affine(projected, "proj1.bias", Activation::Relu)?;
        let projected = self.component_mm("audio.proj2.weight", &projected, frames, true)?;
        let projected = self.audio_affine(projected, "proj2.bias", Activation::Linear)?;
        let mut output = self.zeros(frames / 5 * 4096)?;
        // SAFETY: output keeps only complete nonoverlapping groups of five rows.
        unsafe {
            self.module.whisper.whisper_pool(
                &self.stream,
                LaunchConfig::for_num_elems(output.len() as u32),
                &projected,
                &mut output,
            )?;
        }
        Ok(output)
    }

    fn audio_conv<const PLANAR: bool>(
        &self,
        input: &DeviceBuffer<f32>,
        frames: usize,
        channels: usize,
        stride: usize,
        name: &str,
    ) -> Result<Buffer> {
        let rows = frames.div_ceil(stride);
        let mut columns = self.zeros(rows * channels * 3)?;
        // SAFETY: checked mel/hidden extents; the kernel supplies zero convolution padding.
        unsafe {
            self.module.whisper.whisper_im2col::<PLANAR>(
                &self.stream,
                LaunchConfig::for_num_elems(columns.len() as u32),
                frames,
                channels,
                stride,
                input,
                &mut columns,
            )?;
        }
        let projected = self.component_mm(&format!("audio.{name}.weight"), &columns, rows, true)?;
        self.audio_affine(projected, &format!("{name}.bias"), Activation::Gelu)
    }

    fn audio_block(
        &self,
        layer: usize,
        input: &DeviceBuffer<f32>,
        frames: usize,
        chunk_size: usize,
    ) -> Result<Buffer> {
        let prefix = format!("blk.{layer}");
        let normalized = self.audio_norm(&format!("{prefix}.ln1"), input)?;
        let qkv = self.component_mm(
            &format!("audio.{prefix}.qkv.weight"),
            &normalized,
            frames,
            true,
        )?;
        let attention = self.zeros(frames * 1024)?;
        // SAFETY: 16 width-64 heads; fused Q/K/V rows, Q/V biases and disjoint output.
        unsafe {
            self.module.whisper.whisper_attention(
                &self.stream,
                LaunchConfig::for_num_elems((frames * 16 * 32) as u32),
                frames,
                chunk_size,
                &qkv,
                self.floats(&format!("audio.{prefix}.q.bias"))?,
                self.floats(&format!("audio.{prefix}.v.bias"))?,
                attention.cu_deviceptr() as *mut f32,
            )?;
        }
        self.audio_residual_block(&prefix, input, &attention, frames)
    }

    fn audio_residual_block(
        &self,
        prefix: &str,
        input: &DeviceBuffer<f32>,
        attention: &DeviceBuffer<f32>,
        frames: usize,
    ) -> Result<Buffer> {
        let projected = self.component_mm(
            &format!("audio.{prefix}.out.weight"),
            attention,
            frames,
            true,
        )?;
        let projected =
            self.audio_affine(projected, &format!("{prefix}.out.bias"), Activation::Linear)?;
        let residual = self.add(input, &projected)?;
        let normalized = self.audio_norm(&format!("{prefix}.ln2"), &residual)?;
        let up = self.component_mm(
            &format!("audio.{prefix}.up.weight"),
            &normalized,
            frames,
            true,
        )?;
        let up = self.audio_affine(up, &format!("{prefix}.up.bias"), Activation::Gelu)?;
        let down = self.component_mm(&format!("audio.{prefix}.down.weight"), &up, frames, true)?;
        let down = self.audio_affine(down, &format!("{prefix}.down.bias"), Activation::Linear)?;
        self.add(&residual, &down)
    }

    fn audio_norm(&self, name: &str, input: &DeviceBuffer<f32>) -> Result<Buffer> {
        ensure!(
            input.len().is_multiple_of(1024),
            "invalid audio normalization rows"
        );
        let output = self.zeros(input.len())?;
        // SAFETY: validated 1024-value rows and matching loaded normalization vectors.
        unsafe {
            self.module.vision_math.layer_norm(
                &self.stream,
                LaunchConfig::for_num_elems((input.len() / 1024 * 32) as u32),
                1024,
                input.len() / 1024,
                1e-5,
                input,
                self.floats(&format!("audio.{name}.weight"))?,
                self.floats(&format!("audio.{name}.bias"))?,
                output.cu_deviceptr() as *mut f32,
            )?;
        }
        Ok(output)
    }

    fn audio_affine(&self, values: Buffer, name: &str, activation: Activation) -> Result<Buffer> {
        let bias = self.floats(&format!("audio.{name}"))?;
        ensure!(
            !bias.is_empty() && values.len().is_multiple_of(bias.len()),
            "invalid audio affine rows"
        );
        let config = LaunchConfig::for_num_elems(values.len() as u32);
        let pointer = values.cu_deviceptr() as *mut f32;
        // SAFETY: the consumed output lease is unique and bias spans complete rows.
        unsafe {
            match activation {
                Activation::Linear => self.module.vision_math.affine_in_place::<false>(
                    &self.stream,
                    config,
                    values.len(),
                    bias.len(),
                    bias,
                    pointer,
                )?,
                Activation::Gelu => self.module.minicpm_vision.minicpm_gelu_erf::<false>(
                    &self.stream,
                    config,
                    values.len(),
                    bias.len(),
                    bias,
                    pointer,
                )?,
                Activation::Relu => self.module.whisper.whisper_relu(
                    &self.stream,
                    config,
                    values.len(),
                    bias.len(),
                    bias,
                    pointer,
                )?,
            }
        }
        Ok(values)
    }
}
