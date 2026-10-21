use super::*;
use anyhow::{Context, ensure};

impl Engine {
    /// Presence applies only to generated tokens and resets for each request.
    /// Uncommitted speculative prefixes are applied separately and never stored.
    pub fn set_presence_penalty(&mut self, strength: f32) -> Result<()> {
        ensure!(
            strength.is_finite() && (-2.0..=2.0).contains(&strength),
            "invalid presence penalty"
        );
        if strength != 0.0 && self.presence.is_none() {
            self.presence = Some(DeviceBuffer::zeroed(&self.stream, self.vocabulary_size())?);
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
            (token as usize) < self.vocabulary_size(),
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

    pub(super) fn adjusted_logits_prefix(
        &self,
        logits: &DeviceBuffer<f32>,
        prefix: &[u32],
        causal: bool,
    ) -> Result<Option<Buffer>> {
        if self.presence_strength == 0.0 {
            return Ok(None);
        }
        ensure!(
            prefix
                .iter()
                .all(|&t| (t as usize) < self.vocabulary_size()),
            "invalid presence prefix token"
        );
        ensure!(
            !causal || prefix.len() * self.vocabulary_size() == logits.len(),
            "invalid causal prefix extent"
        );
        if !prefix.is_empty() {
            self.prepare_tokens(prefix)?;
        }
        ensure!(
            logits.len().is_multiple_of(self.vocabulary_size()),
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
                &self.token_ids,
                prefix.len(),
                causal,
                self.presence_strength,
                output.cu_deviceptr() as *mut f32,
            )?;
        }
        Ok(Some(output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_penalties_exclude_future_proposals() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.new_stream()?;
        let module = kernels::load(&context)?;
        let logits = DeviceBuffer::from_host(&stream, &[3.0f32; 15])?;
        let history = DeviceBuffer::from_host(&stream, &[0.0f32, 0.75, 0.0, 0.0, 0.0])?;
        let prefix = DeviceBuffer::from_host(&stream, &[2u32, 4, 3])?;
        let output = DeviceBuffer::<f32>::zeroed(&stream, 15)?;
        for causal in [true, false] {
            // SAFETY: three complete vocabulary rows and a three-token prefix;
            // each output index is bounded by the kernel's checked input extent.
            unsafe {
                module.penalty.apply(
                    &stream,
                    LaunchConfig::for_num_elems(15),
                    &logits,
                    &history,
                    &prefix,
                    3,
                    causal,
                    0.75,
                    output.cu_deviceptr() as *mut f32,
                )?;
            }
            let actual = output.to_host_vec(&stream)?;
            for row in 0..3 {
                let end = if causal { row + 1 } else { 3 };
                for token in 0..5 {
                    let seen = token == 1 || [2, 4, 3][..end].contains(&token);
                    assert_eq!(actual[row * 5 + token], if seen { 2.25 } else { 3.0 });
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires the real GGUF model and CUDA"]
    fn speculative_history_contains_only_confirmed_tokens() -> Result<()> {
        let path = std::env::var("LIA_TEST_MODEL").context("set LIA_TEST_MODEL")?;
        let mut engine = Engine::load(&path, 256)?;
        engine.set_presence_penalty(0.75)?;
        engine.forward_many(&[1, 2, 3], false)?;
        assert!(
            engine
                .presence
                .as_ref()
                .unwrap()
                .to_host_vec(&engine.stream)?
                .iter()
                .all(|x| *x == 0.0)
        );
        let mut target = crate::sampling::Sampling::new(1.0, 20, 0.95, 42)?;
        let mut draft = crate::sampling::Sampling::new(2.0, 20, 1.0, 123)?;
        let mut first = engine.current_sample(&mut target)?;
        let mut seen = std::collections::BTreeSet::new();
        let mut rejected = false;
        for step in 0..32 {
            if first == 248046 {
                break;
            }
            let (confirmed, next, proposed, matched) =
                engine.speculate_stochastic(first, 2, 248046, &mut target, &mut draft)?;
            rejected |= matched < proposed;
            seen.extend(confirmed.iter().copied());
            let mask = engine
                .presence
                .as_ref()
                .unwrap()
                .to_host_vec(&engine.stream)?;
            for (token, penalty) in mask.into_iter().enumerate() {
                assert_eq!(
                    penalty,
                    if seen.contains(&(token as u32)) {
                        0.75
                    } else {
                        0.0
                    }
                );
            }
            first = next;
            if step >= 4 && rejected {
                break;
            }
        }
        assert!(
            rejected,
            "exercise at least one rejected speculative suffix"
        );
        engine.set_presence_penalty(0.0)?;
        assert!(
            engine
                .presence
                .as_ref()
                .unwrap()
                .to_host_vec(&engine.stream)?
                .iter()
                .all(|x| *x == 0.0)
        );
        Ok(())
    }
}
