use super::*;
use anyhow::Context;

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
        let (qgate, kraw, v) = if cache_only {
            let [k, v] = self.mm_many(
                [
                    &format!("{prefix}.attn_k.weight"),
                    &format!("{prefix}.attn_v.weight"),
                ],
                x,
            )?;
            (None, k, v)
        } else {
            let [q, k, v] = self.mm_many(
                [
                    &format!("{prefix}.attn_q.weight"),
                    &format!("{prefix}.attn_k.weight"),
                    &format!("{prefix}.attn_v.weight"),
                ],
                x,
            )?;
            (Some(q), k, v)
        };
        let q = self.zeros(batch * 6144)?;
        let k = self.zeros(batch * 1024)?;
        let layout = attention_plan::plan(
            batch,
            position + batch,
            self.attention_mma_decode,
            self.attention_query_tiles,
        );
        let tensor = layout.tensor;
        let span = layout.span;
        let parts = layout.parts;
        let cap_parts = parts;
        let required = layout.queries * 24 * cap_parts * 258;
        let mut workspace = self.attention_work.borrow_mut();
        if workspace
            .as_ref()
            .is_none_or(|buffer| buffer.len() < required)
        {
            // Free the old large allocation before growing, avoiding both
            // retained tail shapes and transient double allocation peaks.
            anyhow::ensure!(
                !self.preallocated,
                "reserved attention workspace extent exceeded"
            );
            workspace.take();
            *workspace = Some(
                DeviceBuffer::zeroed(&self.stream, required).with_context(|| {
                    format!("allocating attention workspace of {required} floats")
                })?,
            );
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
            // SAFETY: the persistent table holds MAX_PREFILL_BATCH complete 64-dimensional
            // rotations, matching the maximum validated prefill batch.
            unsafe {
                let launch = LaunchConfig::for_num_elems((batch * 32) as u32);
                if let Some(plan) = &rotary.plan {
                    self.module.position_table.table(
                        &self.stream,
                        launch,
                        batch,
                        position,
                        plan.len(),
                        plan.next_text_position(),
                        self.rope_base,
                        &rotary.positions,
                        rotary.data.cu_deviceptr() as *mut f32,
                    )?;
                } else {
                    self.module.rotary.rope_table(
                        &self.stream,
                        launch,
                        64,
                        batch,
                        position,
                        self.rope_base,
                        rotary.data.cu_deviceptr() as *mut f32,
                    )?;
                }
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
            for query_start in (0..batch).step_by(layout.queries) {
                let query_count = (batch - query_start).min(layout.queries);
                if tensor {
                    macro_rules! tensor_attention {
                        ($grouped:literal, $heads:expr, $queries:expr) => {{
                            self.module.flash_mma.flash_mma::<$grouped>(
                                &self.stream,
                                LaunchConfig {
                                    grid_dim: (
                                        ($queries.div_ceil(16) * $heads * parts) as u32,
                                        1,
                                        1,
                                    ),
                                    block_dim: (256, 1, 1),
                                    shared_mem_bytes: 0,
                                },
                                24,
                                4,
                                tokens,
                                span,
                                parts,
                                query_start,
                                &q,
                                kc,
                                vc,
                                partial.cu_deviceptr() as *mut f32,
                            )?;
                        }};
                    }
                    if batch <= 8 {
                        tensor_attention!(true, 4, (batch * 6));
                    } else {
                        tensor_attention!(false, 24, query_count);
                    }
                } else if span == 64 {
                    attention!(64);
                } else {
                    attention!(128);
                }
                self.module.attention.flash_merge::<256>(
                    &self.stream,
                    LaunchConfig::for_num_elems((query_count * 24 * 32) as u32),
                    query_count * 24,
                    parts,
                    cap_parts,
                    query_start * 24,
                    partial,
                    qgate
                        .as_ref()
                        .ok_or_else(|| anyhow!("missing attention query"))?,
                    attended.cu_deviceptr() as *mut f32,
                )?;
            }
        }
        Ok(Some(
            self.mm(&format!("{prefix}.attn_output.weight"), &attended)?,
        ))
    }
}
