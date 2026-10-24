//! CAM++ input fbank: 16 kHz, 25/10 ms, Povey window, 80 Kaldi mel bins.
use anyhow::{Result, ensure};
mod fft;
pub struct Features {
    pub frames: usize,
    /// Time-major rows of 80 log-energy values.
    pub values: Vec<f32>,
}
pub struct Processor {
    window: [f32; 400],
    filters: Vec<f32>,
    transform: fft::Transform,
}
impl Default for Processor {
    fn default() -> Self {
        let window = std::array::from_fn(|i| {
            let angle = i as f32 * (core::f64::consts::TAU / 399.) as f32;
            (0.5 - 0.5 * angle.cos()).powf(0.85)
        });
        let low_precise = 1127. * (1. + 20. / 700.0f64).ln();
        let low = low_precise as f32;
        let delta = ((1127. * (1. + 8000. / 700.0f64).ln() - low_precise) / 81.) as f32;
        let frequencies: Vec<f32> = (0..256)
            .map(|i| 1127. * (1. + (i as f32 * 31.25) / 700.).ln())
            .collect();
        let mut filters = Vec::with_capacity(80 * 256);
        for bin in 0..80 {
            let left = low + bin as f32 * delta;
            let center = low + (bin as f32 + 1.) * delta;
            let right = low + (bin as f32 + 2.) * delta;
            filters.extend(frequencies.iter().map(|mel| {
                ((mel - left) / (center - left))
                    .min((right - mel) / (right - center))
                    .max(0.)
            }));
        }
        Self {
            window,
            filters,
            transform: fft::Transform::default(),
        }
    }
}
impl Processor {
    pub fn extract(&self, samples: &[f32]) -> Result<Features> {
        ensure!(
            (400..=4_800_000).contains(&samples.len()) && samples.iter().all(|v| v.is_finite()),
            "invalid Kaldi fbank PCM input"
        );
        let frames = 1 + (samples.len() - 400) / 160;
        let mut values = Vec::with_capacity(frames * 80);
        for frame in 0..frames {
            let input = &samples[frame * 160..frame * 160 + 400];
            let windowed = self.windowed(input);
            let power = self.transform.power(&windowed);
            for filter in self.filters.chunks_exact(256) {
                let mut energy = 0.0f32;
                for (&weight, &power) in filter.iter().zip(&power[..256]) {
                    energy = weight.mul_add(power, energy);
                }
                values.push(energy.max(f32::EPSILON).ln());
            }
        }
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "nonfinite Kaldi fbank output"
        );
        Ok(Features { frames, values })
    }
    fn windowed(&self, input: &[f32]) -> [f32; 400] {
        let mean = (input.iter().map(|v| f64::from(*v)).sum::<f64>() / 400.) as f32;
        let centered: [f32; 400] = std::array::from_fn(|i| input[i] - mean);
        std::array::from_fn(|i| {
            (centered[i] - 0.97 * centered[i.saturating_sub(1)]) * self.window[i]
        })
    }
    /// Remove each mel channel's utterance mean, as required by the speaker model.
    pub fn centered(&self, samples: &[f32]) -> Result<Features> {
        let mut features = self.extract(samples)?;
        for bin in 0..80 {
            let mean = (features
                .values
                .chunks_exact(80)
                .map(|row| f64::from(row[bin]))
                .sum::<f64>()
                / features.frames as f64) as f32;
            for row in features.values.chunks_exact_mut(80) {
                row[bin] -= mean;
            }
        }
        Ok(features)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires external Kaldi spectral-stage oracle"]
    fn spectral_parameters_and_frame_match_reference() -> Result<()> {
        let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_KALDI_ORACLE")?);
        let read = |name: &str| -> Result<Vec<f32>> {
            Ok(std::fs::read(root.join(name))?
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect())
        };
        let processor = Processor::default();
        for (name, actual, limit) in [
            ("window", processor.window.to_vec(), 0.000002),
            ("filters", processor.filters.clone(), 0.0001),
        ] {
            let expected = read(&format!("{name}.f32"))?;
            ensure!(
                actual.len() == expected.len(),
                "spectral parameter extent mismatch"
            );
            let max = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let unequal = actual
                .iter()
                .zip(&expected)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            println!("{name}: max_abs={max}, unequal={unequal}");
            ensure!(max < limit, "spectral parameter mismatch");
            std::fs::write(
                root.join(format!("{name}.native.f32")),
                actual
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
        }
        let input = read("reference-voice-input.f32")?;
        let windowed = processor.windowed(&input[445 * 160..445 * 160 + 400]);
        let power = processor.transform.power(&windowed);
        for (name, actual) in [("windowed", windowed.to_vec()), ("power", power.to_vec())] {
            let expected = read(&format!("frame445-{name}.f32"))?;
            let max = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("frame445 {name}: max_abs={max}");
            std::fs::write(
                root.join(format!("frame445-{name}.native.f32")),
                actual
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            ensure!(max < 0.001, "spectral frame mismatch");
        }
        Ok(())
    }
    #[test]
    fn frame_edges_silence_and_invalid_samples() -> Result<()> {
        let processor = Processor::default();
        for (count, frames) in [(400, 1), (559, 1), (560, 2), (16000, 98)] {
            let features = processor.centered(&vec![0.; count])?;
            assert_eq!(features.frames, frames);
            assert!(features.values.iter().all(|v| *v == 0.));
        }
        assert!(processor.extract(&[0.; 399]).is_err());
        assert!(processor.extract(&[f32::NAN; 400]).is_err());
        Ok(())
    }
}
