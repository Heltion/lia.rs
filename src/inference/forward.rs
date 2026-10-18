use super::*;

impl Engine {
    pub fn forward_many(&mut self, tokens: &[u32], want_logits: bool) -> Result<Vec<f32>> {
        match self.forward_many_device(tokens, want_logits)? {
            Some(logits) => Ok(logits.to_host_vec(&self.stream)?),
            None => Ok(Vec::new()),
        }
    }
    pub(super) fn forward_many_device(
        &mut self,
        tokens: &[u32],
        want_logits: bool,
    ) -> Result<Option<Buffer>> {
        if tokens.is_empty()
            || tokens.len() > crate::gpu::MAX_PREFILL_BATCH
            || self.position + tokens.len() > self.capacity
        {
            return Err(anyhow!("invalid forward batch or exhausted context"));
        }
        self.prepare_tokens(tokens)?;
        let capture = self.begin_graph()?;
        let key = (0, tokens.len(), want_logits, self.capture);
        let result = self.forward_many_inner(tokens, want_logits);
        self.finish_graph(capture, key, result)
    }
    fn forward_many_inner(&mut self, tokens: &[u32], want_logits: bool) -> Result<Option<Buffer>> {
        let batch = tokens.len();
        let mut hidden = self.embed_many(tokens)?;
        for layer in 0..64 {
            let normalized = self.norm(&format!("blk.{layer}.attn_norm.weight"), &hidden, 5120)?;
            let attention = if (layer + 1) % 4 == 0 {
                self.full_attention(layer, &normalized)?
            } else {
                self.linear_attention(layer, &normalized)?
            };
            drop(normalized);
            let residual = self.add(&hidden, &attention)?;
            drop(hidden);
            drop(attention);
            let normalized = self.norm(
                &format!("blk.{layer}.post_attention_norm.weight"),
                &residual,
                5120,
            )?;
            let [up, gate] = self.mm_many(
                [
                    &format!("blk.{layer}.ffn_up.weight"),
                    &format!("blk.{layer}.ffn_gate.weight"),
                ],
                &normalized,
            )?;
            let activated = self.activate_owned(up, &gate)?;
            drop(gate);
            drop(normalized);
            let down = self.mm(&format!("blk.{layer}.ffn_down.weight"), &activated)?;
            drop(activated);
            hidden = self.add(&residual, &down)?;
        }
        let normalized = self.norm("output_norm.weight", &hidden, 5120)?;
        drop(hidden);
        let logits = if want_logits {
            Some(self.mm("output.weight", &normalized)?)
        } else {
            None
        };
        self.mtp_many(tokens, &normalized)?;
        let mut last = self.zeros(5120)?;
        // SAFETY: a complete final token row is copied to a fresh hidden buffer.
        unsafe {
            self.module.basic.last_hidden(
                &self.stream,
                LaunchConfig::for_num_elems(5120),
                5120,
                &normalized,
                &mut last,
            )?;
        }
        self.last_hidden = Some(last);
        if self.capture {
            self.verify_hidden = Some(normalized);
        }
        self.position += batch;
        self.mtp_prepared = None;
        Ok(logits)
    }
    pub fn forward(&mut self, token: u32) -> Result<Vec<f32>> {
        self.step(token, true)
    }
    pub(super) fn step(&mut self, token: u32, want_logits: bool) -> Result<Vec<f32>> {
        if self.position >= self.capacity {
            return Err(anyhow!("context capacity exceeded"));
        }
        if self.mtp_prepared != Some((self.position, token)) {
            self.mtp(token, false)?;
        }
        self.mtp_prepared = Some((self.position, token));
        let capture = self.begin_graph()?;
        let result = self.step_device(token, want_logits);
        let output = self.finish_graph(capture, (1, 1, want_logits, self.capture), result)?;
        let logits = output
            .map(|x| x.to_host_vec(&self.stream))
            .transpose()?
            .unwrap_or_default();
        anyhow::ensure!(
            logits.iter().all(|x| x.is_finite()),
            "nonfinite model logits"
        );
        Ok(logits)
    }
    fn step_device(&mut self, token: u32, want_logits: bool) -> Result<Option<Buffer>> {
        let mut hidden = self.embed(token)?;
        for layer in 0..64 {
            let normalized = self.norm(&format!("blk.{layer}.attn_norm.weight"), &hidden, 5120)?;
            let attention = if (layer + 1) % 4 == 0 {
                self.full_attention(layer, &normalized)?
            } else {
                self.linear_attention(layer, &normalized)?
            };
            drop(normalized);
            let residual = self.add(&hidden, &attention)?;
            drop(hidden);
            drop(attention);
            let normalized = self.norm(
                &format!("blk.{layer}.post_attention_norm.weight"),
                &residual,
                5120,
            )?;
            let [up, gate] = self.mm_many(
                [
                    &format!("blk.{layer}.ffn_up.weight"),
                    &format!("blk.{layer}.ffn_gate.weight"),
                ],
                &normalized,
            )?;
            let activated = self.activate_owned(up, &gate)?;
            drop(gate);
            drop(normalized);
            let down = self.mm(&format!("blk.{layer}.ffn_down.weight"), &activated)?;
            drop(activated);
            hidden = self.add(&residual, &down)?;
        }
        let normalized = self.norm("output_norm.weight", &hidden, 5120)?;
        drop(hidden);
        let logits = if want_logits {
            Some(self.mm("output.weight", &normalized)?)
        } else {
            None
        };
        self.last_hidden = Some(normalized);
        self.position += 1;
        Ok(logits)
    }
}
