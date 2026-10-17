use super::*;

impl Engine {
    pub(super) fn draft_chain(&mut self, first: u32, maximum: usize, eos: u32) -> Result<Vec<u32>> {
        let mut proposals = Vec::new();
        let mut token = first;
        let mut chain: Option<Buffer> = None;
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
            token = self.greedy(&logits)?[0];
            chain = Some(hidden);
            proposals.push(token);
            if token == eos {
                break;
            }
        }
        self.mtp_prepared = None;
        Ok(proposals)
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
        if maximum == 0 || first == eos {
            return Err(anyhow!("invalid speculative iteration"));
        }
        let maximum = maximum.min(self.capacity - self.position);
        if maximum == 0 {
            return Err(anyhow!("context capacity exceeded"));
        }
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
        let start_position = self.position;
        let drafts = self.draft_chain(first, maximum.saturating_sub(1).min(self.max_draft), eos)?;
        let mut inputs = vec![first];
        inputs.extend_from_slice(&drafts);
        self.capture = true;
        let verification = self.forward_many_device(&inputs, true);
        self.capture = false;
        let verified = verification?.ok_or_else(|| anyhow!("missing verification logits"))?;
        let verified = self.greedy(&verified)?;
        let mut accepted = 0;
        for (index, proposal) in drafts.iter().enumerate() {
            if verified[index] != *proposal {
                break;
            }
            accepted += 1;
            if *proposal == eos {
                break;
            }
        }
        let count = 1 + accepted;
        let logits = verified[accepted];
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
        self.verify_hidden = None;
        Ok((inputs, logits, drafts.len(), accepted))
    }
}
