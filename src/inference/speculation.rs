use super::*;
use crate::sampling::{Distribution, Sampling};

impl Engine {
    pub(super) fn prepare_speculation_buffers(&mut self) -> Result<()> {
        if self.spec_states.is_empty() {
            for state in &self.states {
                self.spec_states.push(match state {
                    State::Recurrent { conv, matrix } => Some((
                        DeviceBuffer::zeroed(&self.stream, (self.max_draft + 1) * conv.len())?,
                        DeviceBuffer::zeroed(&self.stream, (self.max_draft + 1) * matrix.len())?,
                    )),
                    State::Attention { .. } => None,
                });
            }
        }
        Ok(())
    }

    pub(super) fn draft_chain(
        &mut self,
        first: u32,
        maximum: usize,
        eos: u32,
        mut sampling: Option<&mut Sampling>,
    ) -> Result<(Vec<u32>, Vec<Distribution>)> {
        let mut proposals = Vec::new();
        let mut distributions = Vec::new();
        let mut token = first;
        let mut chain: Option<Buffer> = None;
        let mut prefix = vec![first];
        for step in 0..maximum {
            if self.position + step >= self.capacity {
                break;
            }
            let h: &DeviceBuffer<f32> = match &chain {
                Some(h) => h,
                None => self
                    .last_hidden
                    .as_ref()
                    .ok_or_else(|| anyhow!("missing target hidden state"))?,
            };
            let (hidden, logits) = self.mtp_given(token, h, self.position + step, true)?;
            let logits = logits.ok_or_else(|| anyhow!("missing draft logits"))?;
            token = if let Some(sampling) = sampling.as_deref_mut() {
                let distribution = self
                    .distributions_prefix(&logits, sampling, &prefix, false)?
                    .pop()
                    .ok_or_else(|| anyhow!("missing draft distribution"))?;
                let token = sampling.draw(&distribution);
                distributions.push(distribution);
                token
            } else {
                self.greedy_prefix(&logits, &prefix, false)?[0]
            };
            chain = Some(hidden);
            proposals.push(token);
            prefix.push(token);
            if token == eos {
                break;
            }
        }
        self.mtp_prepared = None;
        Ok((proposals, distributions))
    }
    /// Verify the configured native MTP proposals in one target batch. Rejected
    /// suffixes restore the captured recurrent state at the accepted position.
    /// The returned logits belong to the last accepted target state.
    pub fn speculate(
        &mut self,
        first: u32,
        maximum: usize,
        eos: u32,
    ) -> Result<(Vec<u32>, u32, usize, usize)> {
        self.speculate_inner(first, maximum, eos, None, None)
    }
    pub fn speculate_sampled(
        &mut self,
        first: u32,
        maximum: usize,
        eos: u32,
        sampling: &mut crate::sampling::Sampling,
    ) -> Result<(Vec<u32>, u32, usize, usize)> {
        self.speculate_inner(first, maximum, eos, Some(sampling), None)
    }
    pub fn speculate_stochastic(
        &mut self,
        first: u32,
        maximum: usize,
        eos: u32,
        sampling: &mut Sampling,
        draft: &mut Sampling,
    ) -> Result<(Vec<u32>, u32, usize, usize)> {
        self.speculate_inner(first, maximum, eos, Some(sampling), Some(draft))
    }

    fn speculate_inner(
        &mut self,
        first: u32,
        maximum: usize,
        eos: u32,
        sampling: Option<&mut crate::sampling::Sampling>,
        draft_sampling: Option<&mut Sampling>,
    ) -> Result<(Vec<u32>, u32, usize, usize)> {
        if maximum == 0 || first == eos {
            return Err(anyhow!("invalid speculative iteration"));
        }
        let maximum = maximum.min(self.capacity - self.position);
        if maximum == 0 {
            return Err(anyhow!("context capacity exceeded"));
        }
        if !self.geometry.mtp {
            self.step(first, false)?;
            self.remember_generated(first)?;
            let next = if let Some(sampling) = sampling {
                self.current_sample(sampling)?
            } else {
                self.current_token()?
            };
            return Ok((vec![first], next, 0, 0));
        }
        self.prepare_speculation_buffers()?;
        let start_position = self.position;
        let stochastic = draft_sampling.is_some();
        let (drafts, draft_distributions) = self.draft_chain(
            first,
            maximum.saturating_sub(1).min(self.max_draft),
            eos,
            draft_sampling,
        )?;
        let mut inputs = vec![first];
        inputs.extend_from_slice(&drafts);
        self.capture = true;
        let verification = self.forward_many_device(&inputs, true);
        self.capture = false;
        let verified = verification?.ok_or_else(|| anyhow!("missing verification logits"))?;
        let (accepted, logits) = if stochastic {
            let sampling = sampling.ok_or_else(|| anyhow!("missing target sampler"))?;
            let distributions = self.distributions_prefix(&verified, sampling, &inputs, true)?;
            sampling.select_speculative(&distributions, &drafts, &draft_distributions, eos)?
        } else {
            self.select_verified(&verified, &drafts, eos, sampling, &inputs)?
        };
        let count = 1 + accepted;
        if count < inputs.len() {
            for (state, captured) in self.states.iter().zip(&self.spec_states) {
                if let (State::Recurrent { conv, matrix }, Some((c, m))) = (state, captured) {
                    // SAFETY: each captured allocation contains a complete state per verified position;
                    // accepted is an index into the verified batch of at most eight.
                    unsafe {
                        self.module.basic.restore_position(
                            &self.stream,
                            LaunchConfig::for_num_elems(conv.len() as u32),
                            accepted * conv.len(),
                            conv.len(),
                            c,
                            conv.cu_deviceptr() as *mut f32,
                        )?;
                        self.module.basic.restore_position(
                            &self.stream,
                            LaunchConfig::for_num_elems(matrix.len() as u32),
                            accepted * matrix.len(),
                            matrix.len(),
                            m,
                            matrix.cu_deviceptr() as *mut f32,
                        )?;
                    }
                }
            }
            let hidden = self
                .verify_hidden
                .as_ref()
                .ok_or_else(|| anyhow!("missing verified hidden states"))?;
            let last = self.zeros(5120)?;
            // SAFETY: the selected complete hidden row belongs to an accepted token.
            unsafe {
                self.module.basic.restore_position(
                    &self.stream,
                    LaunchConfig::for_num_elems(5120),
                    accepted * 5120,
                    5120,
                    hidden,
                    last.cu_deviceptr() as *mut f32,
                )?;
            }
            self.last_hidden = Some(last);
            self.position = start_position + count;
            self.mtp_prepared = None;
        }
        inputs.truncate(count);
        for &token in &inputs {
            self.remember_generated(token)?;
        }
        self.verify_hidden = None;
        Ok((inputs, logits, drafts.len(), accepted))
    }
}
