//! Compose native reference WAV decoding, resampling, S3, CAM++ and acoustic mel.
use super::*;
use crate::models::minicpm_o45::{PreparedVoice, audio, flow_mel, reference, resample::Resampler};
impl Engine {
    pub fn load_reference_voice_weights(&mut self, directory: &str) -> Result<()> {
        self.load_s3(directory)?;
        self.load_speaker(directory)
    }
    pub fn prepare_voice_wav(&self, bytes: &[u8], seconds: f64) -> Result<PreparedVoice> {
        let waveform = reference::normalize_wav(bytes, seconds)?;
        let samples_16k = Resampler::new(24000, 16000)?.process(&waveform)?;
        let features =
            audio::Processor::for_s3().extract(&samples_16k, audio::Normalization::Dynamic)?;
        let codes = self.s3_codes(&features)?;
        let embedding = self.speaker_embedding_pcm(&samples_16k)?;
        let acoustic = flow_mel::Processor::default().extract(&waveform)?;
        ensure!(
            !acoustic.values.is_empty(),
            "empty acoustic reference features"
        );
        let mut mel = acoustic.values;
        let target = codes.len() * 2 * 80;
        if mel.len() > target {
            mel.truncate(target);
        } else if mel.len() < target {
            let tail = mel[mel.len() - 80..].to_vec();
            while mel.len() < target {
                mel.extend_from_slice(&tail);
            }
        }
        let result = PreparedVoice {
            codes,
            mel,
            embedding,
        };
        result.voice()?;
        Ok(result)
    }
}
