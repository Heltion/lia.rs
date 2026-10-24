//! Released duplex system context includes the reference voice as audio input.
use crate::models::minicpm_o45::{
    audio::{Features, Processor},
    resample::Resampler,
    wav,
};
use anyhow::{Context, Result, ensure};

pub fn system_texts(instructions: &str) -> (String, String) {
    let instructions = if instructions.is_empty() {
        "Streaming Omni Conversation."
    } else {
        instructions
    };
    (
        format!("<|im_start|>system\n{instructions}\n<|audio_start|>"),
        "<|audio_end|><|im_end|>".into(),
    )
}

pub fn reference_features(bytes: &[u8]) -> Result<Features> {
    let audio = wav::decode(bytes).context("decoding duplex reference WAV")?;
    let samples = Resampler::new(audio.sample_rate, 16000)?
        .process(&audio.samples)
        .context("resampling duplex reference to 16 kHz")?;
    let usable = samples.len() / 1600 * 1600;
    ensure!(
        (1600..=480000).contains(&usable),
        "duplex reference requires 0.1..=30 seconds"
    );
    Processor::default()
        .extract_padded(&samples[..usable])
        .context("preparing duplex reference mel features")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn system_context_places_audio_inside_system_boundaries() {
        let (prefix, suffix) = system_texts("");
        assert_eq!(
            prefix,
            "<|im_start|>system\nStreaming Omni Conversation.\n<|audio_start|>"
        );
        assert_eq!(suffix, "<|audio_end|><|im_end|>");
        assert!(
            system_texts("Repeat the user.")
                .0
                .contains("system\nRepeat the user.\n")
        );
    }
    #[test]
    fn reference_is_trimmed_to_complete_audio_tokens() -> Result<()> {
        let bytes = wav::pcm16(&vec![0.0; 3217], 16000)?;
        let features = reference_features(&bytes)?;
        assert_eq!(features.frames, 20);
        assert_eq!(features.values.len(), 80 * 20);
        assert!(features.values.iter().all(|v| v.is_finite()));
        assert!(reference_features(&wav::pcm16(&[0.0; 1599], 16000)?).is_err());
        assert!(reference_features(b"invalid WAV").is_err());
        Ok(())
    }
}
