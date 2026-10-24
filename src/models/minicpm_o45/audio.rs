//! Native Whisper-compatible centered STFT and Slaney log-mel preprocessing.
use anyhow::{Result, ensure};
pub mod stream;
pub mod units;

pub enum Normalization {
    Dynamic,
    Fixed,
}

pub struct Features {
    pub frames: usize,
    /// Mel-major layout; the processor selects 80 Whisper or 128 S3 rows.
    pub values: Vec<f32>,
}

#[derive(Clone)]
pub struct Processor {
    bins: usize,
    window: Vec<f64>,
    real: Vec<f64>,
    imaginary: Vec<f64>,
    filters: Vec<f64>,
}

impl Default for Processor {
    fn default() -> Self {
        Self::with_bins(80)
    }
}

impl Processor {
    /// S3 v2 uses the same STFT and normalization with 128 Slaney mel filters.
    pub fn for_s3() -> Self {
        Self::with_bins(128)
    }

    fn with_bins(bins: usize) -> Self {
        let tau = std::f64::consts::TAU;
        let mut real = Vec::with_capacity(201 * 400);
        let mut imaginary = Vec::with_capacity(201 * 400);
        for frequency in 0..201 {
            for sample in 0..400 {
                let angle = tau * frequency as f64 * sample as f64 / 400.0;
                real.push(angle.cos());
                imaginary.push(-angle.sin());
            }
        }
        let upper = 15.0 + (8.0f64).ln() * 27.0 / (6.4f64).ln();
        let frequencies: Vec<f64> = (0..bins + 2)
            .map(|i| {
                let mel = upper * i as f64 / (bins + 1) as f64;
                if mel < 15.0 {
                    mel * 200.0 / 3.0
                } else {
                    1000.0 * ((mel - 15.0) * (6.4f64).ln() / 27.0).exp()
                }
            })
            .collect();
        let mut filters = Vec::with_capacity(bins * 201);
        for mel in 0..bins {
            let left = frequencies[mel];
            let center = frequencies[mel + 1];
            let right = frequencies[mel + 2];
            for frequency in 0..201 {
                let hz = frequency as f64 * 40.0;
                let weight = ((hz - left) / (center - left))
                    .min((right - hz) / (right - center))
                    .max(0.0);
                filters.push(weight * 2.0 / (right - left));
            }
        }
        Self {
            bins,
            window: (0..400)
                .map(|i| 0.5 - 0.5 * (tau * i as f64 / 400.0).cos())
                .collect(),
            real,
            imaginary,
            filters,
        }
    }
}

impl Processor {
    /// Match ordinary checkpoint requests: zero-pad to 30 s before STFT, then
    /// retain the ceil-divided valid frame count from the attention mask.
    pub fn extract_padded(&self, samples: &[f32]) -> Result<Features> {
        ensure!(
            !samples.is_empty() && samples.len() <= 480_000,
            "audio must contain 1..=480000 samples at 16 kHz"
        );
        let mut padded = samples.to_vec();
        padded.resize(480_000, 0.0);
        let all = self.extract(&padded, Normalization::Dynamic)?;
        let frames = samples.len().div_ceil(160);
        let mut values = Vec::with_capacity(self.bins * frames);
        for mel in 0..self.bins {
            values.extend_from_slice(&all.values[mel * all.frames..mel * all.frames + frames]);
        }
        Ok(Features { frames, values })
    }

    /// Input is finite mono PCM at 16 kHz, before any model-specific chunking.
    pub fn extract(&self, samples: &[f32], normalization: Normalization) -> Result<Features> {
        ensure!(
            !samples.is_empty() && samples.len() <= 480_000,
            "audio must contain 1..=480000 samples at 16 kHz"
        );
        ensure!(
            samples.iter().all(|x| x.is_finite()),
            "audio contains non-finite samples"
        );
        let frames = samples.len() / 160;
        ensure!(frames > 0, "audio is shorter than one mel frame");
        let mut values = vec![0.0f32; self.bins * frames];
        let mut buffer = [0.0f64; 400];
        let mut powers = [0.0f64; 201];
        for frame in 0..frames {
            for (i, value) in buffer.iter_mut().enumerate() {
                let signed = (frame * 160 + i) as isize - 200;
                let index = if samples.len() == 1 {
                    0
                } else {
                    let period = 2 * (samples.len() - 1);
                    let position = signed.rem_euclid(period as isize) as usize;
                    position.min(period - position)
                };
                *value = f64::from(samples[index]) * self.window[i];
            }
            for (frequency, power) in powers.iter_mut().enumerate() {
                let mut re = 0.0;
                let mut im = 0.0;
                for (i, value) in buffer.iter().enumerate() {
                    re += value * self.real[frequency * 400 + i];
                    im += value * self.imaginary[frequency * 400 + i];
                }
                // The official NumPy path stores FFT bins as complex64 first.
                let re = f64::from(re as f32);
                let im = f64::from(im as f32);
                *power = re * re + im * im;
            }
            for mel in 0..self.bins {
                let energy: f64 = powers
                    .iter()
                    .enumerate()
                    .map(|(frequency, power)| power * self.filters[mel * 201 + frequency])
                    .sum();
                values[mel * frames + frame] = energy.max(1e-10).log10() as f32;
            }
        }
        let floor = match normalization {
            Normalization::Dynamic => {
                values.iter().copied().fold(f32::NEG_INFINITY, f32::max) - 8.0
            }
            Normalization::Fixed => -10.0,
        };
        for value in &mut values {
            *value = (value.max(floor) + 4.0) / 4.0;
        }
        Ok(Features { frames, values })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_and_invalid_pcm() -> Result<()> {
        let processor = Processor::default();
        for normalization in [Normalization::Dynamic, Normalization::Fixed] {
            let features = processor.extract(&vec![0.0; 16000], normalization)?;
            assert_eq!(features.frames, 100);
            assert_eq!(features.values, vec![-1.5; 8000]);
        }
        assert!(processor.extract(&[], Normalization::Dynamic).is_err());
        assert!(
            processor
                .extract(&[0.0; 159], Normalization::Dynamic)
                .is_err()
        );
        assert!(
            processor
                .extract(&vec![0.0; 480001], Normalization::Fixed)
                .is_err()
        );
        let mut invalid = vec![0.0; 160];
        invalid[7] = f32::NAN;
        assert!(processor.extract(&invalid, Normalization::Fixed).is_err());
        invalid[7] = f32::INFINITY;
        assert!(processor.extract(&invalid, Normalization::Fixed).is_err());
        Ok(())
    }
}
