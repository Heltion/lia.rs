use super::*;

impl Engine {
    pub(super) fn full_attention(&self, layer: usize, x: &DeviceBuffer<f32>) -> Result<Buffer> {
        self.full_attention_at(layer, x, self.position, false)?
            .ok_or_else(|| anyhow!("missing attention output"))
    }
    pub(super) fn full_attention_at(
        &self,
        layer: usize,
        x: &DeviceBuffer<f32>,
        position: usize,
        cache_only: bool,
    ) -> Result<Option<Buffer>> {
        let batch = x.len() / 5120;
        let prefix = format!("blk.{layer}");
        let qgate = if cache_only {
            None
        } else {
            Some(self.mm(&format!("{prefix}.attn_q.weight"), x)?)
        };
        let kraw = self.mm(&format!("{prefix}.attn_k.weight"), x)?;
        let v = self.mm(&format!("{prefix}.attn_v.weight"), x)?;
        let q = self.zeros(batch * 6144)?;
        let k = self.zeros(batch * 1024)?;
        let tile = if batch <= 8 { 64 } else { 128 };
        let parts = (position + batch).div_ceil(tile);
        let cap_parts = parts;
        let required = batch * 24 * cap_parts * 258;
        let mut workspace = self.attention_work.borrow_mut();
        if workspace
            .as_ref()
            .is_none_or(|buffer| buffer.len() < required)
        {
            // Free the old large allocation before growing, avoiding both
            // retained tail shapes and transient double allocation peaks.
            workspace.take();
            *workspace = Some(DeviceBuffer::zeroed(&self.stream, required)?);
        }
        let partial = workspace
            .as_ref()
            .ok_or_else(|| anyhow!("missing attention workspace"))?;
        let attended = self.zeros(batch * 6144)?;
        let State::Attention { k: kc, v: vc } = &self.states[layer] else {
            return Err(anyhow!("invalid attention layer state"));
        };
        let tokens = position + batch;
        let mut rotary = self.rotary.borrow_mut();
        if rotary.key != Some((position, batch)) {
            // SAFETY: the persistent table holds 512 complete 64-dimensional
            // rotations, matching the maximum validated prefill batch.
            unsafe {
                self.module.rotary.rope_table(
                    &self.stream,
                    LaunchConfig::for_num_elems((batch * 32) as u32),
                    64,
                    batch,
                    position,
                    self.rope_base,
                    rotary.data.cu_deviceptr() as *mut f32,
                )?;
            }
            rotary.key = Some((position, batch));
        }

        // SAFETY: dimension and extent checks at load, bounded position, matching
        // Q/KV head grouping, and disjoint warp/element output ownership.
        unsafe {
            if let Some(qgate) = &qgate {
                self.module.rotary.norm_rope_cached(
                    &self.stream,
                    LaunchConfig::for_num_elems((batch * 24 * 32) as u32),
                    256,
                    24,
                    512,
                    64,
                    self.eps,
                    qgate,
                    self.floats(&format!("{prefix}.attn_q_norm.weight"))?,
                    &rotary.data,
                    q.cu_deviceptr() as *mut f32,
                )?;
            }
            self.module.rotary.norm_rope_cached(
                &self.stream,
                LaunchConfig::for_num_elems((batch * 4 * 32) as u32),
                256,
                4,
                256,
                64,
                self.eps,
                &kraw,
                self.floats(&format!("{prefix}.attn_k_norm.weight"))?,
                &rotary.data,
                k.cu_deviceptr() as *mut f32,
            )?;
            self.module.rotary.cache_kv_half(
                &self.stream,
                LaunchConfig::for_num_elems((batch * 1024) as u32),
                1024,
                position,
                &k,
                &v,
                kc.cu_deviceptr() as *mut u16,
                vc.cu_deviceptr() as *mut u16,
            )?;
            if cache_only {
                return Ok(None);
            }
            macro_rules! attention {
                ($tile:literal) => {{
                    self.module.flash_half.flash_partial_half::<256, $tile>(
                        &self.stream,
                        LaunchConfig::for_num_elems((batch * 24 * parts * 32) as u32),
                        24,
                        4,
                        tokens,
                        parts,
                        cap_parts,
                        &q,
                        kc,
                        vc,
                        partial.cu_deviceptr() as *mut f32,
                    )?;
                }};
            }
            if tile == 64 {
                attention!(64);
            } else {
                attention!(128);
            }
            self.module.attention.flash_merge::<256>(
                &self.stream,
                LaunchConfig::for_num_elems((batch * 24 * 32) as u32),
                batch * 24,
                parts,
                cap_parts,
                partial,
                qgate
                    .as_ref()
                    .ok_or_else(|| anyhow!("missing attention query"))?,
                attended.cu_deviceptr() as *mut f32,
            )?;
        }
        Ok(Some(
            self.mm(&format!("{prefix}.attn_output.weight"), &attended)?,
        ))
    }
}
