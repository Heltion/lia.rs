//! Typed acoustic conditioning; preprocessing of a reference waveform is separate.
use anyhow::{Result, ensure};
pub const MAX_CODEC_CODES: usize = 4096;
pub const MAX_MEL_FRAMES: usize = MAX_CODEC_CODES * 2;
pub struct Voice<'a> {
    codes: &'a [u32],
    mel: &'a [f32],
    embedding: &'a [f32],
}
impl<'a> Voice<'a> {
    pub fn new(codes: &'a [u32], mel: &'a [f32], embedding: &'a [f32]) -> Result<Self> {
        ensure!(
            codes.len() < MAX_CODEC_CODES
                && codes.iter().all(|&v| v < 6561)
                && mel.len() == codes.len() * 2 * 80
                && embedding.len() == 192
                && mel.iter().chain(embedding).all(|v| v.is_finite()),
            "invalid acoustic voice conditioning"
        );
        Ok(Self {
            codes,
            mel,
            embedding,
        })
    }
    pub fn codes(&self) -> &[u32] {
        self.codes
    }
    pub fn mel(&self) -> &[f32] {
        self.mel
    }
    pub fn embedding(&self) -> &[f32] {
        self.embedding
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn voice_requires_matching_acoustic_frames_and_finite_embedding() {
        assert!(Voice::new(&[1, 2], &[0.; 320], &[0.; 192]).is_ok());
        assert!(Voice::new(&[], &[], &[0.; 192]).is_ok());
        assert!(Voice::new(&[1], &[0.; 80], &[0.; 192]).is_err());
        assert!(Voice::new(&[6561], &[0.; 160], &[0.; 192]).is_err());
        assert!(Voice::new(&[], &[], &[f32::NAN; 192]).is_err());
    }
}

/// Owned conditioning prepared directly from a reference waveform.
pub struct PreparedVoice {
    pub(crate) codes: Vec<u32>,
    pub(crate) mel: Vec<f32>,
    pub(crate) embedding: Vec<f32>,
}
impl PreparedVoice {
    pub fn voice(&self) -> Result<Voice<'_>> {
        Voice::new(&self.codes, &self.mel, &self.embedding)
    }
}
