//! Centered small-window source analysis and overlap-add waveform synthesis.
use super::*;
impl Engine {
    pub(super) fn hift_forward_spectrum(&self, source: &DeviceBuffer<f32>) -> Result<Buffer> {
        ensure!(
            source.len() > 8 && source.len().is_multiple_of(4),
            "invalid HiFT excitation extent"
        );
        let mut output = self.zeros((source.len() / 4 + 1) * 18)?;
        // SAFETY: reflection padding is bounded by the validated input length.
        unsafe {
            self.module.hift_spectral.hift_stft(
                &self.stream,
                LaunchConfig::for_num_elems(output.len() as u32),
                source,
                &mut output,
            )?;
        }
        Ok(output)
    }
    pub(in crate::inference::codec) fn hift_inverse(
        &self,
        spectrum: &DeviceBuffer<f32>,
    ) -> Result<Buffer> {
        ensure!(
            spectrum.len().is_multiple_of(18) && spectrum.len() > 18,
            "invalid HiFT spectral extent"
        );
        let frames = spectrum.len() / 18;
        let mut output = self.zeros((frames - 1) * 4)?;
        // SAFETY: a 16-point periodic Hann window at hop four has a positive retained envelope.
        unsafe {
            self.module.hift_spectral.hift_istft(
                &self.stream,
                LaunchConfig::for_num_elems(output.len() as u32),
                frames,
                spectrum,
                &mut output,
            )?;
        }
        Ok(output)
    }
}
