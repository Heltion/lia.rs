//! Attention preparation keeps the fused and original arithmetic selectable.
use super::*;
impl Engine {
    pub(super) fn qwen3_prepare(
        &self,
        prefix: &str,
        layer: usize,
        q: &DeviceBuffer<f32>,
        k: &DeviceBuffer<f32>,
        v: &DeviceBuffer<f32>,
    ) -> Result<Buffer> {
        let State::Attention { k: kc, v: vc } = &self.states[layer] else {
            anyhow::bail!("Qwen3 requires attention cache");
        };
        if self.qwen3_prepare_fusion {
            let qw = self.floats(&format!("{prefix}.attn_q_norm.weight"))?;
            let kw = self.floats(&format!("{prefix}.attn_k_norm.weight"))?;
            anyhow::ensure!(
                qw.len() == 128 && kw.len() == 128,
                "invalid Qwen3 head norm weights"
            );
            let output = self.zeros(q.len())?;
            let batch = q.len() / 4096;
            // SAFETY: caller validates complete Q/K/V heads and cache extent;
            // each warp writes one disjoint head with the original RMS tree.
            unsafe {
                self.module.qwen3_prepare.qwen3_prepare(
                    &self.stream,
                    LaunchConfig::for_num_elems(u32::try_from(batch * 40 * 32)?),
                    batch,
                    self.position,
                    self.rope_base,
                    self.eps,
                    q,
                    k,
                    v,
                    qw,
                    kw,
                    output.cu_deviceptr() as *mut f32,
                    kc.cu_deviceptr() as *mut u16,
                    vc.cu_deviceptr() as *mut u16,
                )?;
            }
            return Ok(output);
        }
        let q = self.norm(&format!("{prefix}.attn_q_norm.weight"), &q, 128)?;
        let k = self.norm(&format!("{prefix}.attn_k_norm.weight"), &k, 128)?;
        let rotated_q = self.zeros(q.len())?;
        let rotated_k = self.zeros(k.len())?;
        let rotary = self.rotary.borrow();
        // SAFETY: complete validated Q/K/V heads and bounded cache writes.
        unsafe {
            self.module.llama.llama_rope(
                &self.stream,
                LaunchConfig::for_num_elems(q.len() as u32),
                32,
                self.position,
                self.rope_base,
                false,
                &rotary.positions,
                &q,
                rotated_q.cu_deviceptr() as *mut f32,
            )?;
            self.module.llama.llama_rope(
                &self.stream,
                LaunchConfig::for_num_elems(k.len() as u32),
                8,
                self.position,
                self.rope_base,
                false,
                &rotary.positions,
                &k,
                rotated_k.cu_deviceptr() as *mut f32,
            )?;
            self.module.dense_attention.dense_store_kv(
                &self.stream,
                LaunchConfig::for_num_elems(k.len() as u32),
                self.position * 1024,
                &rotated_k,
                &v,
                kc.cu_deviceptr() as *mut u16,
                vc.cu_deviceptr() as *mut u16,
            )?;
        }
        Ok(rotated_q)
    }
}
