//! MiniCPM-o token-to-mel upsample encoder; waveform synthesis is a later stage.
use super::*;
use anyhow::ensure;
mod cfm;
mod hift;
mod ops;
mod pipeline;
pub(super) mod stream;
mod weights;
use ops::Activation;

impl Engine {
    /// Return time-major 80-channel acoustic features at twice the code rate.
    pub fn codec_encoder_features(&self, codes: &[u32]) -> Result<Vec<f32>> {
        self.codec_encode(codes)?
            .to_host_vec(&self.stream)
            .map_err(Into::into)
    }
    fn codec_encode(&self, codes: &[u32]) -> Result<Buffer> {
        let embedded = self.codec_embedded(codes)?;
        let looked = self.codec_conv(
            "encoder.pre_lookahead_layer.conv1",
            &embedded,
            codes.len(),
            4,
            0,
            1,
            Activation::Leaky,
        )?;
        let looked = self.codec_conv(
            "encoder.pre_lookahead_layer.conv2",
            &looked,
            codes.len(),
            3,
            2,
            1,
            Activation::Linear,
        )?;
        let mut hidden = self.add(&embedded, &looked)?;
        let positions = self.codec_positions(codes.len())?;
        for layer in 0..6 {
            hidden = self.codec_block(
                &format!("encoder.encoders.{layer}"),
                &hidden,
                &positions,
                codes.len(),
            )?;
        }
        let up = self.codec_conv(
            "encoder.up_layer.conv",
            &hidden,
            codes.len(),
            5,
            4,
            2,
            Activation::Linear,
        )?;
        let frames = codes.len() * 2;
        let mut hidden = self.codec_embed("encoder.up_embed", &up, frames)?;
        let positions = self.codec_positions(frames)?;
        for layer in 0..4 {
            hidden = self.codec_block(
                &format!("encoder.up_encoders.{layer}"),
                &hidden,
                &positions,
                frames,
            )?;
        }
        let normalized = self.codec_norm("encoder.after_norm", &hidden, 1e-5)?;
        self.codec_linear("encoder_proj", &normalized, frames, Activation::Linear)
    }
    fn codec_embedded(&self, codes: &[u32]) -> Result<Buffer> {
        ensure!(
            self.geometry.qwen3 && self.weights.contains_key("codec.encoder_proj.weight"),
            "codec encoder weights are not loaded"
        );
        ensure!(
            !codes.is_empty()
                && codes.len() <= crate::models::minicpm_o45::MAX_CODEC_CODES
                && codes.iter().all(|&code| code < 6561),
            "invalid codec code sequence"
        );
        let weight = self.weight("codec.input_embedding.weight")?;
        let ids = DeviceBuffer::from_host(&self.stream, codes)?;
        let mut embedded = self.zeros(codes.len() * 512)?;
        // SAFETY: validated vocabulary IDs and complete 512-channel Float32 rows.
        unsafe {
            self.module.basic.embedding_many(
                &self.stream,
                LaunchConfig::for_num_elems(embedded.len() as u32),
                0,
                4,
                1,
                512,
                &ids,
                &weight.data,
                &mut embedded,
            )?;
        }
        self.codec_embed("encoder.embed", &embedded, codes.len())
    }
    fn codec_positions(&self, frames: usize) -> Result<Buffer> {
        let mut positions = self.zeros((frames * 2 - 1) * 512)?;
        // SAFETY: positive sequence length and complete positive/negative positional rows.
        unsafe {
            self.module.codec.codec_positions(
                &self.stream,
                LaunchConfig::for_num_elems(positions.len() as u32),
                frames,
                &mut positions,
            )?;
        }
        Ok(positions)
    }
    fn codec_embed(&self, name: &str, input: &DeviceBuffer<f32>, frames: usize) -> Result<Buffer> {
        let projected =
            self.codec_linear(&format!("{name}.out.0"), input, frames, Activation::Linear)?;
        let normalized = self.codec_norm(&format!("{name}.out.1"), &projected, 1e-5)?;
        // SAFETY: this lease is uniquely owned and all complete rows are scaled in place.
        unsafe {
            self.module.codec.codec_scale(
                &self.stream,
                LaunchConfig::for_num_elems(normalized.len() as u32),
                normalized.len(),
                512.0f32.sqrt(),
                normalized.cu_deviceptr() as *mut f32,
            )?;
        }
        Ok(normalized)
    }
    fn codec_block(
        &self,
        name: &str,
        input: &DeviceBuffer<f32>,
        positions: &DeviceBuffer<f32>,
        frames: usize,
    ) -> Result<Buffer> {
        let normalized = self.codec_norm(&format!("{name}.norm_mha"), input, 1e-12)?;
        let qkv = self.codec_linear(
            &format!("{name}.qkv"),
            &normalized,
            frames,
            Activation::Linear,
        )?;
        let pos = self.component_mm(
            &format!("codec.{name}.pos.weight"),
            positions,
            frames * 2 - 1,
            true,
        )?;
        let attention = self.zeros(frames * 512)?;
        // SAFETY: full eight width-64 heads, matching positional projection and bias rows.
        unsafe {
            self.module.codec.codec_relative_attention(
                &self.stream,
                LaunchConfig::for_num_elems((frames * 8 * 32) as u32),
                frames,
                &qkv,
                &pos,
                self.floats(&format!("codec.{name}.pos_bias_u"))?,
                self.floats(&format!("codec.{name}.pos_bias_v"))?,
                attention.cu_deviceptr() as *mut f32,
            )?;
        }
        let projected = self.codec_linear(
            &format!("{name}.out"),
            &attention,
            frames,
            Activation::Linear,
        )?;
        let residual = self.add(input, &projected)?;
        let normalized = self.codec_norm(&format!("{name}.norm_ff"), &residual, 1e-12)?;
        let up = self.codec_linear(&format!("{name}.up"), &normalized, frames, Activation::Silu)?;
        let down = self.codec_linear(&format!("{name}.down"), &up, frames, Activation::Linear)?;
        self.add(&residual, &down)
    }
}
