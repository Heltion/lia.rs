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
        anyhow::ensure!(
            self.paging.is_none(),
            "use forward_paged after configuring paging"
        );
        if tokens.is_empty()
            || tokens.len() > crate::gpu::MAX_PREFILL_BATCH
            || self.position + tokens.len() > self.capacity
        {
            return Err(anyhow!("invalid forward batch or exhausted context"));
        }
        self.decode_replay.borrow_mut().take();
        self.qwen3_decode_replay.borrow_mut().take();
        self.prepare_tokens(tokens)?;
        let capture = self.begin_graph()?;
        let key = (0, tokens.len(), want_logits, self.capture);
        let result = self.forward_many_inner(tokens, want_logits);
        self.finish_graph(capture, key, result)
    }
    fn forward_many_inner(&mut self, tokens: &[u32], want_logits: bool) -> Result<Option<Buffer>> {
        let hidden = self.embed_many(tokens)?;
        self.forward_hidden(tokens, hidden, want_logits)
    }
    /// Append model-produced embeddings to the same causal language KV history
    /// used by token forwards. Returns row-major logits when requested.
    pub fn forward_embeddings(&mut self, values: &[f32], want_logits: bool) -> Result<Vec<f32>> {
        anyhow::ensure!(
            self.geometry.qwen3 && !self.geometry.mtp && self.paging.is_none(),
            "direct embedding feed requires an unpaged Qwen3 decoder without MTP"
        );
        anyhow::ensure!(
            !values.is_empty()
                && values.len().is_multiple_of(self.geometry.hidden)
                && values.iter().all(|v| v.is_finite()),
            "invalid direct embedding rows"
        );
        let batch = values.len() / self.geometry.hidden;
        anyhow::ensure!(
            batch <= crate::gpu::MAX_PREFILL_BATCH && batch <= self.capacity - self.position,
            "direct embedding feed exceeds batch or context capacity"
        );
        let mut hidden = self.zeros(values.len())?;
        hidden.copy_from_host(&self.stream, values)?;
        let tokens = vec![0; batch];
        self.decode_replay.borrow_mut().take();
        self.qwen3_decode_replay.borrow_mut().take();
        self.prepare_tokens(&tokens)?;
        let capture = self.begin_graph()?;
        let key = (7, batch, want_logits, self.capture);
        let result = self.forward_hidden(&tokens, hidden, want_logits);
        let result = self.finish_graph(capture, key, result)?;
        result
            .map(|v| v.to_host_vec(&self.stream))
            .transpose()
            .map(|v| v.unwrap_or_default())
            .map_err(Into::into)
    }
    fn forward_hidden(
        &mut self,
        tokens: &[u32],
        mut hidden: Buffer,
        want_logits: bool,
    ) -> Result<Option<Buffer>> {
        let batch = tokens.len();
        for layer in 0..self.geometry.layers {
            let normalized = self.norm(
                &format!("blk.{layer}.attn_norm.weight"),
                &hidden,
                self.geometry.hidden,
            )?;
            let attention = if self.geometry.llama {
                self.llama_attention(layer, &normalized)?
            } else if self.geometry.qwen3 {
                self.qwen3_attention(layer, &normalized)?
            } else if (layer + 1) % 4 == 0 {
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
                self.geometry.hidden,
            )?;
            let down = self.feed_forward(layer, &normalized)?;
            drop(normalized);
            hidden = self.add(&residual, &down)?;
        }
        let normalized = self.norm("output_norm.weight", &hidden, self.geometry.hidden)?;
        drop(hidden);
        let logits = if want_logits {
            Some(self.mm("output.weight", &normalized)?)
        } else {
            None
        };
        self.mtp_many(tokens, &normalized)?;
        let mut last = self.zeros(self.geometry.hidden)?;
        // SAFETY: a complete final token row is copied to a fresh hidden buffer.
        unsafe {
            self.module.basic.last_hidden(
                &self.stream,
                LaunchConfig::for_num_elems(self.geometry.hidden as u32),
                self.geometry.hidden,
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
        anyhow::ensure!(
            self.paging.is_none(),
            "use forward_paged after configuring paging"
        );
        anyhow::ensure!(
            (token as usize) < self.vocabulary_size(),
            "invalid vocabulary token"
        );
        if self.position >= self.capacity {
            return Err(anyhow!("context capacity exceeded"));
        }
        if self.geometry.mtp && self.mtp_prepared != Some((self.position, token)) {
            self.mtp(token, false)?;
        }
        self.mtp_prepared = Some((self.position, token));
        if self.replay_qwen3_decode(token, want_logits)?
            || self.replay_decode(token, want_logits)?
        {
            return Ok(Vec::new());
        }
        let _scratch = self.activate_qwen3_decode(want_logits);
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
        for layer in 0..self.geometry.layers {
            let normalized = self.norm(
                &format!("blk.{layer}.attn_norm.weight"),
                &hidden,
                self.geometry.hidden,
            )?;
            let attention = if self.geometry.llama {
                self.llama_attention(layer, &normalized)?
            } else if self.geometry.qwen3 {
                self.qwen3_attention(layer, &normalized)?
            } else if (layer + 1) % 4 == 0 {
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
                self.geometry.hidden,
            )?;
            let down = self.feed_forward(layer, &normalized)?;
            drop(normalized);
            hidden = self.add(&residual, &down)?;
        }
        let normalized = self.norm("output_norm.weight", &hidden, self.geometry.hidden)?;
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
