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
        if tokens.is_empty() || tokens.len() > 512 || self.position + tokens.len() > self.capacity {
            return Err(anyhow!("invalid forward batch or exhausted context"));
        }
        let batch = tokens.len();
        let mut hidden = self.embed_many(tokens)?;
        for layer in 0..64 {
            let normalized = self.norm(&format!("blk.{layer}.attn_norm.weight"), &hidden, 5120)?;
            let attention = if (layer + 1) % 4 == 0 {
                self.full_attention(layer, &normalized)?
            } else {
                self.linear_attention(layer, &normalized)?
            };
            let residual = self.add(&hidden, &attention)?;
            let normalized = self.norm(
                &format!("blk.{layer}.post_attention_norm.weight"),
                &residual,
                5120,
            )?;
            let up = self.mm(&format!("blk.{layer}.ffn_up.weight"), &normalized)?;
            let gate = self.mm(&format!("blk.{layer}.ffn_gate.weight"), &normalized)?;
            let activated = self.silu_mul(&up, &gate)?;
            let down = self.mm(&format!("blk.{layer}.ffn_down.weight"), &activated)?;
            hidden = self.add(&residual, &down)?;
        }
        let normalized = self.norm("output_norm.weight", &hidden, 5120)?;
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
        let mut hidden = self.embed(token)?;
        for layer in 0..64 {
            let normalized = self.norm(&format!("blk.{layer}.attn_norm.weight"), &hidden, 5120)?;
            let attention = if (layer + 1) % 4 == 0 {
                self.full_attention(layer, &normalized)?
            } else {
                self.linear_attention(layer, &normalized)?
            };
            let residual = self.add(&hidden, &attention)?;
            let normalized = self.norm(
                &format!("blk.{layer}.post_attention_norm.weight"),
                &residual,
                5120,
            )?;
            let up = self.mm(&format!("blk.{layer}.ffn_up.weight"), &normalized)?;
            let gate = self.mm(&format!("blk.{layer}.ffn_gate.weight"), &normalized)?;
            let activated = self.silu_mul(&up, &gate)?;
            let down = self.mm(&format!("blk.{layer}.ffn_down.weight"), &activated)?;
            hidden = self.add(&residual, &down)?;
        }
        let normalized = self.norm("output_norm.weight", &hidden, 5120)?;
        let logits = if want_logits {
            self.mm("output.weight", &normalized)?
                .to_host_vec(&self.stream)?
        } else {
            Vec::new()
        };
        self.last_hidden = Some(normalized);
        self.position += 1;
        if logits.iter().any(|x| !x.is_finite()) {
            return Err(anyhow!("nonfinite model logits"));
        }
        Ok(logits)
    }
}
