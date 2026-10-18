use super::*;

impl Engine {
    pub(super) fn embed_many(&self, tokens: &[u32]) -> Result<Buffer> {
        if tokens.is_empty()
            || tokens.len() > crate::gpu::MAX_PREFILL_BATCH
            || tokens.iter().any(|t| *t >= 248320)
        {
            return Err(anyhow!("invalid token batch"));
        }
        let w = self.weight("token_embd.weight")?;

        let mut out = self.zeros(tokens.len() * 5120)?;
        // SAFETY: validated token indices, row-aligned embedding table and
        // exactly one guarded output per token embedding element.
        unsafe {
            self.module.basic.embedding_many(
                &self.stream,
                LaunchConfig::for_num_elems((tokens.len() * 5120) as u32),
                w.kind,
                w.bytes,
                w.block,
                5120,
                &self.token_ids,
                &w.data,
                &mut out,
            )?;
        }
        Ok(out)
    }
    pub(super) fn mtp_many(&self, tokens: &[u32], target: &DeviceBuffer<f32>) -> Result<()> {
        let batch = tokens.len();
        let embedding = self.embed_many(tokens)?;
        let prev: &DeviceBuffer<f32> = self.last_hidden.as_deref().unwrap_or(&self.zero_hidden);
        let mut shifted = self.zeros(batch * 5120)?;
        // SAFETY: the first shifted row comes from the previous target state,
        // later rows from this batch's preceding states; all extents match.
        unsafe {
            self.module.basic.shift_hidden(
                &self.stream,
                LaunchConfig::for_num_elems((batch * 5120) as u32),
                5120,
                prev,
                target,
                &mut shifted,
            )?;
        }
        let en = self.norm("blk.64.nextn.enorm.weight", &embedding, 5120)?;
        drop(embedding);
        let hn = self.norm("blk.64.nextn.hnorm.weight", &shifted, 5120)?;
        drop(shifted);
        let mut concatenated = self.zeros(batch * 10240)?;
        // SAFETY: corresponding token rows are concatenated into disjoint rows.
        unsafe {
            self.module.basic.concat_rows(
                &self.stream,
                LaunchConfig::for_num_elems((batch * 10240) as u32),
                5120,
                &en,
                &hn,
                &mut concatenated,
            )?;
        }
        drop(en);
        drop(hn);
        let projected = self.mm("blk.64.nextn.eh_proj.weight", &concatenated)?;
        drop(concatenated);
        let normalized = self.norm("blk.64.attn_norm.weight", &projected, 5120)?;
        drop(projected);
        // Known-prefix priming only needs this block's K/V cache. The MTP
        // attention output and FFN have no persistent state and are unused.
        self.full_attention_at(64, &normalized, self.position, true)?;
        Ok(())
    }
}
