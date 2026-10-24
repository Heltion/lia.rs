//! Reference-voice flow features: 24 kHz, 1920-point Hann STFT, 480-hop, 80 log-mel bins.
use anyhow::{Result, ensure};
mod fft;
pub struct Features {
    pub frames: usize,
    /// Time-major acoustic frames, each with 80 values.
    pub values: Vec<f32>,
}
pub struct Processor {
    window: [f32; 1920],
    filters: Vec<f32>,
    fft: fft::Transform,
}
impl Default for Processor {
    fn default() -> Self {
        let upper = 15. + (8.0f64).ln() * 27. / (6.4f64).ln();
        let frequencies: Vec<f32> = (0..82)
            .map(|i| {
                let mel = upper * i as f64 / 81.;
                (if mel < 15. {
                    mel * 200. / 3.
                } else {
                    1000. * ((mel - 15.) * (6.4f64).ln() / 27.).exp()
                }) as f32
            })
            .collect();
        let mut filters = Vec::with_capacity(80 * 961);
        for bin in 0..80 {
            let left = frequencies[bin];
            let center = frequencies[bin + 1];
            let right = frequencies[bin + 2];
            for frequency in 0..961 {
                let hz = frequency as f32 * 12.5;
                filters.push(
                    ((hz - left) / (center - left))
                        .min((right - hz) / (right - center))
                        .max(0.)
                        * (2. / (right - left)),
                );
            }
        }
        Self {
            window: std::array::from_fn(|i| {
                0.5 - 0.5 * (i as f32 * (core::f64::consts::TAU / 1920.) as f32).cos()
            }),
            filters,
            fft: fft::Transform::default(),
        }
    }
}
impl Processor {
    pub fn extract(&self, samples: &[f32]) -> Result<Features> {
        ensure!(
            (721..=720_000).contains(&samples.len()) && samples.iter().all(|v| v.is_finite()),
            "flow mel requires finite 24 kHz PCM longer than reflection padding and at most 30 seconds"
        );
        let frames = samples.len() / 480;
        let mut values = Vec::with_capacity(frames * 80);
        let mut windowed = [0.; 1920];
        let mut spectrum = [fft::Complex::default(); 1920];
        let mut magnitudes = [0.; 961];
        for frame in 0..frames {
            for (i, x) in windowed.iter_mut().enumerate() {
                let at = (frame * 480 + i) as isize - 720;
                let index = if at < 0 {
                    (-at) as usize
                } else if at >= samples.len() as isize {
                    2 * samples.len() - 2 - at as usize
                } else {
                    at as usize
                };
                *x = samples[index] * self.window[i];
            }
            self.fft.forward(&windowed, &mut spectrum);
            for (bin, x) in magnitudes.iter_mut().enumerate() {
                let re = spectrum[bin].re as f32;
                let im = spectrum[bin].im as f32;
                *x = (re * re + im * im + 1e-9).sqrt();
            }
            for bin in 0..80 {
                let mut energy = 0.0f32;
                for (frequency, x) in magnitudes.iter().enumerate() {
                    energy += x * self.filters[bin * 961 + frequency];
                }
                values.push(energy.max(1e-5).ln());
            }
        }
        Ok(Features { frames, values })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checked_lengths_and_silent_floor() -> Result<()> {
        let processor = Processor::default();
        assert!(processor.extract(&[0.; 720]).is_err());
        assert!(processor.extract(&[f32::NAN; 1920]).is_err());
        assert!(processor.extract(&vec![0.; 720_001]).is_err());
        let silence = processor.extract(&[0.; 1920])?;
        assert_eq!(silence.frames, 4);
        assert_eq!(silence.values.len(), 320);
        assert!(
            silence
                .values
                .iter()
                .all(|v| (*v - 1e-5f32.ln()).abs() < 1e-6)
        );
        Ok(())
    }
}
