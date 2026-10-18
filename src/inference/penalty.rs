use super::*;
use anyhow::{Context, ensure};

impl Engine {
    /// Presence applies only to generated tokens and resets for each request.
    /// Speculative decoding requires transactional histories and is rejected.
    pub fn set_presence_penalty(&mut self, strength: f32) -> Result<()> {
        ensure!(
            strength.is_finite() && (-2.0..=2.0).contains(&strength),
            "invalid presence penalty"
        );
        ensure!(
            strength == 0.0 || !self.geometry.mtp,
            "presence penalty with MTP is not implemented"
        );
        if strength != 0.0 && self.presence.is_none() {
            self.presence = Some(DeviceBuffer::zeroed(&self.stream, 248320)?);
        }
        if let Some(values) = &mut self.presence {
            values.zero_async(&self.stream)?;
        }
        self.presence_strength = strength;
        Ok(())
    }

    pub(super) fn remember_generated(&self, token: u32) -> Result<()> {
        if self.presence_strength == 0.0 {
            return Ok(());
        }
        ensure!(
            token < 248320,
            "invalid generated token for presence penalty"
        );
        let values = self
            .presence
            .as_ref()
            .context("missing presence penalty buffer")?;
        // SAFETY: one stream-ordered thread writes a bounded token slot in the
        // persistent allocation; every consumer is queued on the same stream.
        unsafe {
            self.module.penalty.remember(
                &self.stream,
                LaunchConfig::for_num_elems(1),
                token as usize,
                self.presence_strength,
                values.cu_deviceptr() as *mut f32,
            )?;
        }
        Ok(())
    }

    pub(super) fn adjusted_logits(&self, logits: &DeviceBuffer<f32>) -> Result<Option<Buffer>> {
        if self.presence_strength == 0.0 {
            return Ok(None);
        }
        ensure!(
            logits.len().is_multiple_of(248320),
            "invalid presence logit shape"
        );
        let values = self
            .presence
            .as_ref()
            .context("missing presence penalty buffer")?;
        let output = self.zeros(logits.len())?;
        // SAFETY: full vocabulary rows and a complete persistent mask; one
        // guarded thread owns each disjoint output element.
        unsafe {
            self.module.penalty.apply(
                &self.stream,
                LaunchConfig::for_num_elems(logits.len() as u32),
                logits,
                values,
                output.cu_deviceptr() as *mut f32,
            )?;
        }
        Ok(Some(output))
    }
}
