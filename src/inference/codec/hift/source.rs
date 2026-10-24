//! Source excitation assembly, separately seedable from language sampling.
use super::*;
impl Engine {
    /// Generate the HiFT source from per-mel-frame F0 and explicit time-major nine-channel noise.
    pub fn codec_harmonic_source(&self, f0: &[f32], noise: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            self.weights.contains_key("hift.m_source.l_linear.weight"),
            "HiFT source weights are not loaded"
        );
        ensure!(
            !f0.is_empty()
                && f0.len() <= crate::models::minicpm_o45::MAX_MEL_FRAMES
                && noise.len() == f0.len() * 480 * 9
                && f0.iter().all(|v| v.is_finite() && *v >= 0.0)
                && noise.iter().all(|v| v.is_finite()),
            "invalid HiFT F0 or source noise"
        );
        let f0 = DeviceBuffer::from_host(&self.stream, f0)?;
        let noise = DeviceBuffer::from_host(&self.stream, noise)?;
        self.hift_source(&f0, &noise)?
            .to_host_vec(&self.stream)
            .map_err(Into::into)
    }
    pub(in crate::inference::codec) fn hift_source(
        &self,
        f0: &DeviceBuffer<f32>,
        noise: &DeviceBuffer<f32>,
    ) -> Result<Buffer> {
        let phase = self.zeros(f0.len() * 9)?;
        let mut harmonics = self.zeros(noise.len())?;
        // SAFETY: each phase thread owns one harmonic, and sample outputs are disjoint.
        unsafe {
            self.module.hift_source.hift_phase(
                &self.stream,
                LaunchConfig::for_num_elems(9),
                &f0,
                phase.cu_deviceptr() as *mut f32,
            )?;
            self.module.hift_source.hift_harmonics(
                &self.stream,
                LaunchConfig::for_num_elems(harmonics.len() as u32),
                &f0,
                &phase,
                &noise,
                &mut harmonics,
            )?;
        }
        self.hift_linear("m_source.l_linear", &harmonics, f0.len() * 480, 4)
    }
    /// Synthesize an utterance with MT19937 Box–Muller noise, independent of token sampling.
    pub fn codec_synthesize(&self, mel: &[f32], seed: u64) -> Result<Vec<f32>> {
        ensure!(
            self.weights.contains_key("hift.conv_post.weight"),
            "HiFT decoder is not loaded"
        );
        ensure!(
            !mel.is_empty()
                && mel.len().is_multiple_of(80)
                && mel.len() <= crate::models::minicpm_o45::MAX_MEL_FRAMES * 80
                && mel.iter().all(|v| v.is_finite()),
            "invalid synthesis mel frames"
        );
        let frames = mel.len() / 80;
        let noise = crate::sampling::gaussian(seed, frames * 480 * 9);
        self.codec_synthesize_noise(mel, &noise)
    }
    /// Synthesize mel with an explicit time-major nine-channel Gaussian stream.
    pub fn codec_synthesize_noise(&self, mel: &[f32], noise: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            self.weights.contains_key("hift.conv_post.weight"),
            "HiFT decoder is not loaded"
        );
        ensure!(
            !mel.is_empty()
                && mel.len().is_multiple_of(80)
                && mel.len() <= crate::models::minicpm_o45::MAX_MEL_FRAMES * 80
                && noise.len() == mel.len() / 80 * 480 * 9
                && mel.iter().chain(noise).all(|v| v.is_finite()),
            "invalid synthesis mel or noise"
        );
        let noise = DeviceBuffer::from_host(&self.stream, noise)?;
        let mel = DeviceBuffer::from_host(&self.stream, mel)?;
        self.hift_synthesize_device(&mel, &noise)?
            .to_host_vec(&self.stream)
            .map_err(Into::into)
    }
    pub(in crate::inference::codec) fn hift_synthesize_device(
        &self,
        mel: &DeviceBuffer<f32>,
        noise: &DeviceBuffer<f32>,
    ) -> Result<Buffer> {
        ensure!(
            mel.len().is_multiple_of(80)
                && !mel.is_empty()
                && noise.len() == mel.len() / 80 * 480 * 9,
            "invalid device synthesis input"
        );
        let frames = mel.len() / 80;
        let f0 = self.hift_f0(mel, frames)?;
        let source = self.hift_source(&f0, noise)?;
        let spectrum = self.hift_decode(mel, &source, frames)?;
        self.hift_inverse(&spectrum)
    }
}
