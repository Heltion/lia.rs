//! Mono PCM16 output for native generated waveforms.
use anyhow::{Context, Result, ensure};
pub fn pcm16(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    ensure!(
        (1..=192000).contains(&sample_rate) && samples.iter().all(|v| v.is_finite()),
        "invalid generated WAV rate or samples"
    );
    let size = u32::try_from(
        samples
            .len()
            .checked_mul(2)
            .context("generated WAV extent overflow")?,
    )?;
    let riff_size = size
        .checked_add(36)
        .context("generated WAV container overflow")?;
    let mut bytes = Vec::with_capacity(44 + size as usize);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&riff_size.to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&size.to_le_bytes());
    for &sample in samples {
        let value = (sample * 32768.).round().clamp(-32768., 32767.) as i16;
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pcm16_roundtrip_preserves_rate_and_quantization_bound() -> Result<()> {
        let samples = [-1., -0.4, 0., 0.3, 1.];
        let bytes = pcm16(&samples, 24000)?;
        let decoded = super::super::decode(&bytes)?;
        assert_eq!(decoded.sample_rate, 24000);
        assert_eq!(decoded.samples.len(), samples.len());
        for (&a, &b) in decoded.samples.iter().zip(&samples) {
            assert!((a - b).abs() <= 1. / 32768.);
        }
        assert!(pcm16(&[f32::NAN], 24000).is_err());
        assert!(pcm16(&[0.], 0).is_err());
        Ok(())
    }
}
