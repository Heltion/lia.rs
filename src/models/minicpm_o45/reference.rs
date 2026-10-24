//! Match the fixed reference-audio grid and PCM16 materialization used by Code2Wav.
use super::{resample::Resampler, wav};
use anyhow::{Result, ensure};
/// Return normalized mono 24 kHz samples, including the reference PCM16 boundary.
pub fn normalize_wav(bytes: &[u8], seconds: f64) -> Result<Vec<f32>> {
    ensure!(
        seconds.is_finite() && (1.0..=30.0).contains(&seconds),
        "reference window must be between 1 and 30 seconds"
    );
    let audio = wav::decode(bytes)?;
    let mut samples = Resampler::new(audio.sample_rate, 24000)?.process(&audio.samples)?;
    samples.resize((seconds * 24000.) as usize, 0.);
    // The reference writes a WAV with libsndfile's default PCM16 subtype before
    // computing either sample-rate branch. Its float conversion floors, not rounds.
    for x in &mut samples {
        *x = (*x * 32768.).floor().clamp(-32768., 32767.) / 32768.;
    }
    Ok(samples)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalization_has_checked_window_and_materialization_boundary() -> Result<()> {
        let bytes = wav::pcm16(&[0.1, -0.1, 0.5], 24000)?;
        let samples = normalize_wav(&bytes, 1.)?;
        assert_eq!(samples.len(), 24000);
        assert_eq!(&samples[..3], &wav::decode(&bytes)?.samples);
        assert!(samples[3..].iter().all(|v| *v == 0.));
        assert!(normalize_wav(&bytes, 0.).is_err());
        assert!(normalize_wav(&bytes, f64::NAN).is_err());
        assert!(normalize_wav(&[], 6.).is_err());
        Ok(())
    }
}
