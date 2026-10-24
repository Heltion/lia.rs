//! Native CAM++ speaker embeddings with independent CUDA operators and ONNX loading.
use super::*;
use crate::models::minicpm_o45::kaldi;
use anyhow::{Context, ensure};
mod head;
mod ops;
mod reference;
mod weights;
impl Engine {
    /// Raw 192-dimensional speaker embedding from centered time-major Kaldi fbank.
    pub fn speaker_embedding(&self, features: &kaldi::Features) -> Result<Vec<f32>> {
        ensure!(
            self.weights
                .contains_key("speaker.xvector.dense.linear.weight"),
            "speaker weights are not loaded"
        );
        ensure!(
            (3..=3000).contains(&features.frames)
                && features.values.len() == features.frames * 80
                && features.values.iter().all(|v| v.is_finite()),
            "invalid speaker fbank input"
        );
        let mut input = self.zeros(features.values.len())?;
        input.copy_from_host(&self.stream, &features.values)?;
        let head = self.speaker_head(input, features.frames)?;
        let mut hidden = self.speaker_conv(
            "xvector.tdnn.linear",
            &head,
            features.frames,
            320,
            5,
            2,
            1,
            1,
        )?;
        let frames = features.frames.div_ceil(2);
        let mut channels = 128;
        for (stage, layers) in [12, 24, 16].into_iter().enumerate() {
            for layer in 1..=layers {
                let name = format!("xvector.block{}.tdnnd{layer}", stage + 1);
                let normalized =
                    self.speaker_norm(&format!("{name}.nonlinear1.batchnorm"), &hidden, true)?;
                let projected =
                    self.speaker_linear(&format!("{name}.linear1"), &normalized, frames, 1)?;
                let output = self.speaker_cam(
                    &format!("{name}.cam_layer"),
                    &projected,
                    frames,
                    if stage == 0 { 1 } else { 2 },
                )?;
                hidden = self.speaker_concat(&hidden, &output, frames)?;
                channels += 32;
            }
            let name = format!("xvector.transit{}", stage + 1);
            let normalized =
                self.speaker_norm(&format!("{name}.nonlinear.batchnorm"), &hidden, true)?;
            hidden = self.speaker_linear(&format!("{name}.linear"), &normalized, frames, 0)?;
            channels /= 2;
        }
        ensure!(
            channels == 512 && hidden.len() == frames * 512 && frames > 1,
            "invalid speaker statistics input"
        );
        let hidden = self.speaker_relu(&hidden)?;
        let stats = self.zeros(1024)?;
        // SAFETY: each warp reduces one complete feature channel; variance uses N-1 as upstream.
        unsafe {
            self.module.speaker_math.speaker_stats(
                &self.stream,
                LaunchConfig::for_num_elems(512 * 32),
                frames,
                512,
                &hidden,
                stats.cu_deviceptr() as *mut f32,
            )?;
        }
        let projected = self.speaker_linear("xvector.dense.linear", &stats, 1, 0)?;
        let output = self
            .speaker_norm("xvector.dense.nonlinear.batchnorm", &projected, false)?
            .to_host_vec(&self.stream)?;
        ensure!(
            output.len() == 192 && output.iter().all(|v| v.is_finite()),
            "nonfinite or malformed speaker embedding"
        );
        Ok(output)
    }
    pub fn speaker_embedding_pcm(&self, samples_16k: &[f32]) -> Result<Vec<f32>> {
        self.speaker_embedding(&kaldi::Processor::default().centered(samples_16k)?)
    }
    fn speaker_cam(
        &self,
        name: &str,
        input: &DeviceBuffer<f32>,
        frames: usize,
        dilation: usize,
    ) -> Result<Buffer> {
        let local = self.speaker_conv(
            &format!("{name}.linear_local"),
            input,
            frames,
            128,
            3,
            1,
            dilation,
            0,
        )?;
        let segments = frames.div_ceil(100);
        let context = self.zeros(segments * 128)?;
        // SAFETY: every output channel belongs to one bounded segment; final partial segments use their actual count.
        unsafe {
            self.module.speaker_math.speaker_context(
                &self.stream,
                LaunchConfig::for_num_elems((segments * 128 * 32) as u32),
                frames,
                128,
                input,
                context.cu_deviceptr() as *mut f32,
            )?;
        }
        // Context is constant within each 100-frame segment, so project it once per segment.
        let context = self.speaker_linear(&format!("{name}.linear1"), &context, segments, 1)?;
        let gate = self.speaker_linear(&format!("{name}.linear2"), &context, segments, 2)?;
        let mut output = self.zeros(frames * 32)?;
        // SAFETY: one gate row per complete or partial 100-frame segment.
        unsafe {
            self.module.speaker_math.speaker_gate(
                &self.stream,
                LaunchConfig::for_num_elems(output.len() as u32),
                32,
                &local,
                &gate,
                &mut output,
            )?;
        }
        Ok(output)
    }
}
