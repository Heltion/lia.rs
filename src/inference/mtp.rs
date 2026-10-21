use super::*;

impl Engine {
    pub(super) fn mtp_given(
        &self,
        token: u32,
        h: &DeviceBuffer<f32>,
        position: usize,
        logits: bool,
    ) -> Result<(Buffer, Option<Buffer>)> {
        let capture = self.begin_graph()?;
        let result = self.mtp_given_inner(token, h, position, logits);
        self.finish_graph(capture, (2, 1, logits, self.capture), result)
    }
    fn mtp_given_inner(
        &self,
        token: u32,
        h: &DeviceBuffer<f32>,
        position: usize,
        logits: bool,
    ) -> Result<(Buffer, Option<Buffer>)> {
        let embedding = self.embed_at(token, position)?;
        let en = self.norm("blk.64.nextn.enorm.weight", &embedding, 5120)?;
        let hn = self.norm("blk.64.nextn.hnorm.weight", h, 5120)?;
        let mut concatenated = self.zeros(10240)?;
        // SAFETY: output exactly covers both nonoverlapping input vectors.
        unsafe {
            self.module.basic.concat(
                &self.stream,
                LaunchConfig::for_num_elems(10240),
                &en,
                &hn,
                &mut concatenated,
            )?;
        }
        let projected = self.mm("blk.64.nextn.eh_proj.weight", &concatenated)?;
        let norm = self.norm("blk.64.attn_norm.weight", &projected, 5120)?;
        let Some(attention) = self.full_attention_at(64, &norm, position, !logits)? else {
            // Priming stores K/V for a known token. Its output is not a draft
            // hidden state and is intentionally discarded by the caller.
            return Ok((projected, None));
        };
        let residual = self.add(&projected, &attention)?;
        let normalized = self.norm("blk.64.post_attention_norm.weight", &residual, 5120)?;
        let [up, gate] = self.mm_many(
            ["blk.64.ffn_up.weight", "blk.64.ffn_gate.weight"],
            &normalized,
        )?;
        let activated = self.activate_owned(up, &gate)?;
        drop(gate);
        drop(normalized);
        let down = self.mm("blk.64.ffn_down.weight", &activated)?;
        drop(activated);
        let hidden = self.add(&residual, &down)?;
        let normalized = self.norm("blk.64.nextn.shared_head_norm.weight", &hidden, 5120)?;
        let result = if logits {
            Some(self.mm("output.weight", &normalized)?)
        } else {
            None
        };
        Ok((normalized, result))
    }
    pub(super) fn mtp(&self, token: u32, logits: bool) -> Result<Option<Vec<f32>>> {
        let h: &DeviceBuffer<f32> = self.last_hidden.as_deref().unwrap_or(&self.zero_hidden);
        self.mtp_given(token, h, self.position, logits)?
            .1
            .map(|x| Ok(x.to_host_vec(&self.stream)?))
            .transpose()
    }
    /// Run the model's built-in MTP block on a known next token and the most
    /// recent target hidden state. The returned distribution predicts one further token.
    pub fn draft(&mut self, next_token: u32) -> Result<Vec<f32>> {
        if self.position >= self.capacity {
            return Err(anyhow!("MTP context capacity exceeded"));
        }
        let result = self
            .mtp(next_token, true)?
            .ok_or_else(|| anyhow!("missing MTP output"))?;
        self.mtp_prepared = Some((self.position, next_token));
        Ok(result)
    }
    pub(super) fn greedy(&self, logits: &DeviceBuffer<f32>) -> Result<Vec<u32>> {
        self.greedy_prefix(logits, &[], false)
    }
    pub(super) fn greedy_prefix(
        &self,
        logits: &DeviceBuffer<f32>,
        prefix: &[u32],
        causal: bool,
    ) -> Result<Vec<u32>> {
        let adjusted = self.adjusted_logits_prefix(logits, prefix, causal)?;
        let logits = adjusted.as_deref().unwrap_or(logits);
        let vocabulary = self.vocabulary_size();
        if logits.is_empty() || !logits.len().is_multiple_of(vocabulary) {
            return Err(anyhow!("invalid greedy logit shape"));
        }
        let parts = vocabulary.div_ceil(1024);
        let count = logits.len() / vocabulary * parts;
        let values = self.zeros(count)?;
        let ids = self
            .greedy_work
            .get(&(logits.len() / vocabulary))
            .ok_or_else(|| anyhow!("greedy batch exceeds reserved workspace"))?;
        // SAFETY: one complete warp per disjoint partition, matching partial
        // value and index arrays. Each input row covers the full vocabulary.
        unsafe {
            self.module.selection.argmax_partial(
                &self.stream,
                LaunchConfig::for_num_elems((count * 32) as u32),
                vocabulary,
                parts,
                logits,
                values.cu_deviceptr() as *mut f32,
                ids.cu_deviceptr() as *mut u32,
            )?;
        }
        let values = values.to_host_vec(&self.stream)?;
        let ids = ids.to_host_vec(&self.stream)?;
        values
            .chunks(parts)
            .zip(ids.chunks(parts))
            .map(|(v, i)| {
                if v.iter().any(|x| !x.is_finite()) {
                    return Err(anyhow!("nonfinite greedy logits"));
                }
                let best = v
                    .iter()
                    .zip(i)
                    .max_by(|a, b| a.0.total_cmp(b.0).then(a.1.cmp(b.1)))
                    .ok_or_else(|| anyhow!("empty greedy partition"))?;
                Ok(*best.1)
            })
            .collect()
    }
}
