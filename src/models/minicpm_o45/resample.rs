//! Rational Hann-windowed sinc interpolation with a six-zero-crossing low-pass filter.
use anyhow::{Context, Result, ensure};
struct Kernel {
    start: isize,
    coefficients: Vec<f32>,
}
pub struct Resampler {
    original_rate: u32,
    input_period: usize,
    output_period: usize,
    kernels: Vec<Kernel>,
}
impl Resampler {
    pub fn new(original_rate: u32, target_rate: u32) -> Result<Self> {
        ensure!(
            (1..=192000).contains(&original_rate) && (1..=192000).contains(&target_rate),
            "unsupported audio resampling rate"
        );
        let (mut a, mut b) = (original_rate, target_rate);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        let input_period = (original_rate / a) as usize;
        let output_period = (target_rate / a) as usize;
        let base = input_period.min(output_period) as f64 * 0.99;
        let radius = 6.0 * input_period as f64 / base;
        let width = radius.ceil() as isize;
        let mut kernels = Vec::with_capacity(output_period);
        for phase in 0..output_period {
            // Match the cached reference kernel's Float32 phase and Float64 grid.
            let offset = (-(phase as f32) / output_period as f32) as f64;
            let center = -offset * input_period as f64;
            let start = ((center - radius).floor() as isize).max(-width);
            let end = ((center + radius).ceil() as isize).min(input_period as isize + width);
            let mut coefficients = Vec::with_capacity((end - start) as usize);
            for tap in start..end {
                let t = ((offset + tap as f64 / input_period as f64) * base).clamp(-6.0, 6.0);
                let window = (t * core::f64::consts::PI / 6.0 / 2.0).cos().powi(2);
                let angle = t * core::f64::consts::PI;
                let sinc = if angle == 0.0 {
                    1.0
                } else {
                    angle.sin() / angle
                };
                coefficients.push((sinc * window * (base / input_period as f64)) as f32);
            }
            kernels.push(Kernel {
                start,
                coefficients,
            });
        }
        Ok(Self {
            original_rate,
            input_period,
            output_period,
            kernels,
        })
    }
    /// Zero-padded filtering retains ceil(input_len * target_rate / original_rate) samples.
    pub fn process(&self, samples: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            samples.len() <= self.original_rate as usize * 300
                && samples.iter().all(|v| v.is_finite()),
            "invalid or oversized resampling input"
        );
        if self.input_period == self.output_period {
            return Ok(samples.to_vec());
        }
        let count = samples
            .len()
            .checked_mul(self.output_period)
            .context("resampling extent overflow")?
            .div_ceil(self.input_period);
        let mut output = Vec::with_capacity(count);
        for index in 0..count {
            let kernel = &self.kernels[index % self.output_period];
            let origin = (index / self.output_period * self.input_period) as isize + kernel.start;
            let mut sum = 0.0f32;
            for (tap, &coefficient) in kernel.coefficients.iter().enumerate() {
                let at = origin + tap as isize;
                if at >= 0 && (at as usize) < samples.len() {
                    sum = coefficient.mul_add(samples[at as usize], sum);
                }
            }
            output.push(sum);
        }
        Ok(output)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn downsampling_suppresses_aliases_and_preserves_dc() -> Result<()> {
        let resampler = Resampler::new(48000, 16000)?;
        let dc = resampler.process(&vec![1.; 4800])?;
        assert_eq!(dc.len(), 1600);
        assert!(dc[32..1568].iter().all(|v| (*v - 1.0).abs() < 0.002));
        let high: Vec<f32> = (0..4800)
            .map(|i| (core::f32::consts::TAU * 12000. * i as f32 / 48000.).sin())
            .collect();
        let low = resampler.process(&high)?;
        let rms = (low[32..1568]
            .iter()
            .map(|v| f64::from(*v).powi(2))
            .sum::<f64>()
            / 1536.)
            .sqrt();
        assert!(rms < 0.005, "aliased RMS={rms}");
        Ok(())
    }
    #[test]
    fn rational_tails_identity_and_invalid_inputs() -> Result<()> {
        for (a, b) in [
            (44100, 16000),
            (16000, 24000),
            (24000, 16000),
            (48000, 24000),
        ] {
            let resampler = Resampler::new(a, b)?;
            for count in [0, 1, 7, 101] {
                assert_eq!(
                    resampler.process(&vec![0.; count])?.len(),
                    (count * b as usize).div_ceil(a as usize)
                );
            }
        }
        let resampler = Resampler::new(16000, 16000)?;
        assert_eq!(
            resampler
                .process(&[-0., 1., -1.])?
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            [-0.0f32, 1., -1.]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert!(Resampler::new(0, 16000).is_err());
        assert!(resampler.process(&[f32::NAN]).is_err());
        Ok(())
    }
}
