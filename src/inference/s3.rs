//! Native S3 v2 25 Hz semantic speech tokenizer for reference-audio conditioning.
use super::*;
use crate::models::minicpm_o45::audio::Features;
use anyhow::ensure;
mod ops;
mod weights;
impl Engine {
    pub fn s3_embeddings(&self, features: &Features) -> Result<Vec<f32>> {
        self.encode_s3(features)?
            .to_host_vec(&self.stream)
            .map_err(Into::into)
    }
    pub fn s3_projection(&self, features: &Features) -> Result<Vec<f32>> {
        let hidden = self.encode_s3(features)?;
        self.s3_linear(
            "quantizer",
            &hidden,
            features.frames.div_ceil(4),
            true,
            false,
        )?
        .to_host_vec(&self.stream)
        .map_err(Into::into)
    }
    pub fn s3_codes(&self, features: &Features) -> Result<Vec<u32>> {
        let hidden = self.encode_s3(features)?;
        let frames = features.frames.div_ceil(4);
        let projected = self.s3_linear("quantizer", &hidden, frames, true, false)?;
        let mut output = DeviceBuffer::from_host(&self.stream, &vec![0u32; frames])?;
        // SAFETY: quantizer supplies eight scalar values for every disjoint output code.
        unsafe {
            self.module.s3.s3_codes(
                &self.stream,
                LaunchConfig::for_num_elems(frames as u32),
                &projected,
                &mut output,
            )?;
        }
        output.to_host_vec(&self.stream).map_err(Into::into)
    }
    fn encode_s3(&self, features: &Features) -> Result<Buffer> {
        ensure!(
            self.weights.contains_key("s3.quantizer.weight"),
            "S3 weights are not loaded"
        );
        ensure!(
            (1..=3000).contains(&features.frames)
                && features.values.len() == features.frames * 128
                && features.values.iter().all(|v| v.is_finite()),
            "invalid S3 128-bin mel input"
        );
        let input = DeviceBuffer::from_host(&self.stream, &features.values)?;
        let first = self.s3_conv::<true>(&input, features.frames, 128, "conv1")?;
        let mut hidden =
            self.s3_conv::<false>(&first, features.frames.div_ceil(2), 1280, "conv2")?;
        let frames = features.frames.div_ceil(4);
        for layer in 0..6 {
            let prefix = format!("blk.{layer}");
            let normalized = self.s3_norm(&format!("{prefix}.ln1"), &hidden)?;
            let q = self.s3_linear(&format!("{prefix}.q"), &normalized, frames, true, false)?;
            let k = self.s3_linear(&format!("{prefix}.k"), &normalized, frames, false, false)?;
            let v = self.s3_linear(&format!("{prefix}.v"), &normalized, frames, true, false)?;
            let mut qk = self.zeros(frames * 2560)?;
            let attention = self.zeros(frames * 1280)?;
            let mut memory = self.zeros(frames * 1280)?;
            // SAFETY: all projections cover complete 20-head width-64 rows; memory weights have 1280*31 values.
            unsafe {
                self.module.s3.s3_rotary(
                    &self.stream,
                    LaunchConfig::for_num_elems(qk.len() as u32),
                    &q,
                    &k,
                    &mut qk,
                )?;
                self.module.s3.s3_attention(
                    &self.stream,
                    LaunchConfig::for_num_elems((frames * 20 * 32) as u32),
                    frames,
                    &qk,
                    &v,
                    attention.cu_deviceptr() as *mut f32,
                )?;
                self.module.s3.s3_memory(
                    &self.stream,
                    LaunchConfig::for_num_elems(memory.len() as u32),
                    frames,
                    &v,
                    self.floats(&format!("s3.{prefix}.fsmn.weight"))?,
                    &mut memory,
                )?;
            }
            let projected =
                self.s3_linear(&format!("{prefix}.out"), &attention, frames, true, false)?;
            let attended = self.add(&projected, &memory)?;
            let residual = self.add(&hidden, &attended)?;
            let normalized = self.s3_norm(&format!("{prefix}.ln2"), &residual)?;
            let up = self.s3_linear(&format!("{prefix}.up"), &normalized, frames, true, true)?;
            let down = self.s3_linear(&format!("{prefix}.down"), &up, frames, true, false)?;
            hidden = self.add(&residual, &down)?;
        }
        Ok(hidden)
    }
}
