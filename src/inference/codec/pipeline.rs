//! Join speech codes, acoustic reference conditioning, flow matching and HiFT.
use super::*;
use crate::models::minicpm_o45::{MAX_CODEC_CODES, Voice};
impl Engine {
    /// Generate time-major mel for new codes; initial flow noise includes the prompt prefix.
    pub fn codec_voice_mel(
        &self,
        codes: &[u32],
        voice: &Voice<'_>,
        noise: &[f32],
        steps: usize,
    ) -> Result<Vec<f32>> {
        self.codec_voice_mel_device(codes, voice, noise, steps)?
            .to_host_vec(&self.stream)
            .map_err(Into::into)
    }
    /// Render speech codes with independently seeded flow and HiFT Gaussian streams.
    pub fn codec_render(
        &self,
        codes: &[u32],
        voice: &Voice<'_>,
        steps: usize,
        seed: u64,
    ) -> Result<Vec<f32>> {
        self.codec_voice_validate(codes, voice)?;
        ensure!(
            self.weights.contains_key("hift.conv_post.weight"),
            "HiFT decoder weights must be loaded"
        );
        let noise = crate::sampling::gaussian(seed, (codes.len() + voice.codes().len()) * 2 * 80);
        let mel = self.codec_voice_mel_device(codes, voice, &noise, steps)?;
        let noise =
            crate::sampling::gaussian(seed ^ 0x9e37_79b9_7f4a_7c15, codes.len() * 2 * 480 * 9);
        let noise = DeviceBuffer::from_host(&self.stream, &noise)?;
        self.hift_synthesize_device(&mel, &noise)?
            .to_host_vec(&self.stream)
            .map_err(Into::into)
    }
    fn codec_voice_validate(&self, codes: &[u32], voice: &Voice<'_>) -> Result<()> {
        ensure!(
            !codes.is_empty()
                && codes.len() + voice.codes().len() <= MAX_CODEC_CODES
                && codes.iter().all(|&v| v < 6561),
            "invalid conditioned speech codes"
        );
        ensure!(
            self.weights.contains_key("codec.encoder_proj.weight")
                && self
                    .weights
                    .contains_key("codec.decoder.estimator.final_layer.linear.weight"),
            "acoustic encoder and flow weights must be loaded"
        );
        Ok(())
    }
    fn codec_voice_mel_device(
        &self,
        codes: &[u32],
        voice: &Voice<'_>,
        noise: &[f32],
        steps: usize,
    ) -> Result<Buffer> {
        self.codec_voice_validate(codes, voice)?;
        let mut all = Vec::with_capacity(codes.len() + voice.codes().len());
        all.extend_from_slice(voice.codes());
        all.extend_from_slice(codes);
        let frames = all.len() * 2;
        ensure!(
            noise.len() == frames * 80 && noise.iter().all(|v| v.is_finite()),
            "invalid conditioned flow noise"
        );
        let norm = (voice
            .embedding()
            .iter()
            .map(|v| f64::from(*v).powi(2))
            .sum::<f64>()
            .sqrt() as f32)
            .max(1e-12);
        let embedding: Vec<f32> = voice.embedding().iter().map(|v| v / norm).collect();
        let embedding = DeviceBuffer::from_host(&self.stream, &embedding)?;
        let speaker =
            self.codec_linear("spk_embed_affine_layer", &embedding, 1, Activation::Linear)?;
        let mu = self.codec_encode(&all)?;
        let mut conditioning = vec![0.0; frames * 80];
        conditioning[..voice.mel().len()].copy_from_slice(voice.mel());
        let conditioning = DeviceBuffer::from_host(&self.stream, &conditioning)?;
        let mut x = self.zeros(noise.len())?;
        x.copy_from_host(&self.stream, noise)?;
        let result = self.cfm_integrate(x, &mu, &speaker, &conditioning, frames, steps)?;
        let output = self.zeros(codes.len() * 2 * 80)?;
        // SAFETY: the validated prompt span is removed, retaining exactly the generated-code frames.
        unsafe {
            self.module.basic.restore_position(
                &self.stream,
                LaunchConfig::for_num_elems(output.len() as u32),
                voice.mel().len(),
                output.len(),
                &result,
                output.cu_deviceptr() as *mut f32,
            )?;
        }
        Ok(output)
    }
}
