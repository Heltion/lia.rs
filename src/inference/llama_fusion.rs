//! Prefill keeps normalized and gated matrix inputs packed between operators.
use super::*;
use anyhow::{Context, ensure};

impl Engine {
    pub fn gated_projection_fusion(&self) -> bool {
        self.geometry.llama && self.llama_fusion && self.llama_gated
    }

    pub fn fused_prefill_enabled(&self) -> bool {
        self.geometry.llama && self.llama_fusion
    }

    pub(super) fn llama_fused_hidden(
        &self,
        tokens: &[u32],
        outputs: &[u32],
    ) -> Result<Option<Buffer>> {
        ensure!(
            tokens.len() > 8,
            "fused packed projections require prefill rows"
        );
        // Packed gate inputs are wider than normalized hidden rows. Reserve the
        // complete extent before any projection can install a smaller workspace.
        {
            let needed = tokens.len() * 6144;
            let mut packed = self.prefill_input.borrow_mut();
            if packed.as_ref().is_none_or(|buffer| buffer.len() < needed) {
                ensure!(
                    !self.preallocated,
                    "reserved fused activation workspace exhausted"
                );
                *packed = Some(
                    DeviceBuffer::zeroed(&self.stream, needed)
                        .context("allocating fused prefill activation workspace")?,
                );
            }
        }
        let mut hidden = self.embed_many(tokens)?;
        self.llama_norm_packed("blk.0.attn_norm.weight", &hidden, None, 0)?;
        for layer in 0..self.geometry.layers {
            let last = layer + 1 == self.geometry.layers;
            let attention =
                self.llama_attention_selected(layer, &hidden, true, last.then_some(outputs))?;
            if layer + 1 == self.geometry.layers {
                // Every row has already written its final-layer KV. Only requested
                // outputs need this layer's token-local feed-forward computation.
                if outputs.is_empty() {
                    return Ok(None);
                }
                let selected = self.gather_paged_hidden(&hidden, outputs)?;
                let selected = self.add(&selected, &attention)?;
                let normalized = self.norm(
                    &format!("blk.{layer}.post_attention_norm.weight"),
                    &selected,
                    2048,
                )?;
                let down = self.feed_forward(layer, &normalized)?;
                return Ok(Some(self.add(&selected, &down)?));
            }
            let residual = self
                .llama_norm_packed(
                    &format!("blk.{layer}.post_attention_norm.weight"),
                    &hidden,
                    Some(&attention),
                    if self.llama_gated {
                        tokens.len() * 3072
                    } else {
                        0
                    },
                )?
                .context("missing fused attention residual")?;
            drop(hidden);
            drop(attention);
            let down = if self.llama_gated {
                self.llama_gated_feed_forward(layer, &residual)?
            } else {
                let gate =
                    self.mm_prepared(&format!("blk.{layer}.ffn_gate.weight"), &residual, true)?;
                let up =
                    self.mm_prepared(&format!("blk.{layer}.ffn_up.weight"), &residual, true)?;
                {
                    let packed = self.prefill_input.borrow();
                    let packed = packed
                        .as_ref()
                        .context("missing packed activation workspace")?;
                    ensure!(
                        gate.len() == up.len() && packed.len() >= gate.len() / 2,
                        "invalid packed SiLU extent"
                    );
                    // SAFETY: complete adjacent pairs, read-only gates and disjoint packed storage.
                    unsafe {
                        self.module.llama_fusion.silu_pack(
                            &self.stream,
                            LaunchConfig::for_num_elems((gate.len() / 2) as u32),
                            &gate,
                            &up,
                            packed.cu_deviceptr() as *mut u32,
                        )?;
                    }
                }
                // The float lease supplies dimensions; this prefill projection reads the packed input.
                let down =
                    self.mm_prepared(&format!("blk.{layer}.ffn_down.weight"), &gate, true)?;
                drop(gate);
                drop(up);
                down
            };
            hidden = self
                .llama_norm_packed(
                    &format!("blk.{}.attn_norm.weight", layer + 1),
                    &residual,
                    Some(&down),
                    0,
                )?
                .context("missing fused MLP residual")?;
        }
        anyhow::bail!("MiniCPM5 has no transformer layers")
    }

    fn llama_gated_feed_forward(
        &self,
        layer: usize,
        residual: &DeviceBuffer<f32>,
    ) -> Result<Buffer> {
        let batch = residual.len() / 2048;
        let gate_name = format!("blk.{layer}.ffn_gate.weight");
        let up_name = format!("blk.{layer}.ffn_up.weight");
        let gate = self.weight(&gate_name)?;
        let up = self.weight(&up_name)?;
        ensure!(
            batch > 8
                && residual.len().is_multiple_of(2048)
                && gate.shape == [2048, 6144]
                && up.shape == gate.shape
                && gate.kind == 30
                && up.kind == 30
                && gate.data.len() == 2048 * 6144 * 2
                && up.data.len() == gate.data.len(),
            "invalid fused BF16 gate/up geometry"
        );
        let activated = self.zeros(batch * 6144)?;
        {
            let packed = self.prefill_input.borrow();
            let packed = packed.as_ref().context("missing packed gated input")?;
            ensure!(packed.len() >= batch * 6144, "packed gated input exhausted");
            // SAFETY: complete aligned BF16 matrices and packed normalization,
            // disjoint full output rows, complete CTAs with padded token tails.
            unsafe {
                macro_rules! launch {
                    ($half:literal, $gate:expr, $up:expr) => {
                        self.module.llama_gated.llama_gated::<$half>(
                            &self.stream,
                            LaunchConfig {
                                grid_dim: ((6144 / 64 * batch.div_ceil(64)) as u32, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            2048,
                            6144,
                            batch,
                            $gate,
                            $up,
                            (packed.cu_deviceptr() + (batch * 6144 * 2) as u64) as *const u32,
                            packed.cu_deviceptr() as *mut u32,
                        )?
                    };
                }
                if let Some((gate, up)) = self
                    .llama_weight_cache
                    .get(&gate_name)
                    .zip(self.llama_weight_cache.get(&up_name))
                {
                    launch!(true, gate, up);
                } else {
                    launch!(false, &gate.data, &up.data);
                }
                // Normalization wrote the narrow input beyond the complete gated
                // output range. Down projection consumes that output in place.
            }
        }
        // The float lease supplies dimensions only; the projection reads packed data.
        self.mm_prepared(&format!("blk.{layer}.ffn_down.weight"), &activated, true)
    }

    fn llama_norm_packed(
        &self,
        name: &str,
        input: &DeviceBuffer<f32>,
        add: Option<&DeviceBuffer<f32>>,
        packed_offset: usize,
    ) -> Result<Option<Buffer>> {
        let weight = self.floats(name)?;
        ensure!(
            input.len().is_multiple_of(2048) && weight.len() == 2048,
            "invalid fused normalization geometry"
        );
        if let Some(add) = add {
            ensure!(add.len() == input.len(), "fused residual extent mismatch");
        }
        let output = add.map(|_| self.zeros(input.len())).transpose()?;
        let packed = self.prefill_input.borrow();
        let packed = packed
            .as_ref()
            .context("missing packed normalization workspace")?;
        ensure!(
            packed_offset <= packed.len() && packed.len() - packed_offset >= input.len() / 2,
            "packed normalization workspace exhausted"
        );
        let config = LaunchConfig {
            grid_dim: ((input.len() / 2048) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: complete2048-element rows and full CTAs. The no-add specialization
        // never writes its unused residual pointer; the add specialization owns a fresh lease.
        unsafe {
            if let Some(add) = add {
                self.module.llama_fusion.norm_pack::<true>(
                    &self.stream,
                    config,
                    input.len() / 2048,
                    self.eps,
                    input,
                    add,
                    weight,
                    output
                        .as_ref()
                        .context("missing residual allocation")?
                        .cu_deviceptr() as *mut f32,
                    (packed.cu_deviceptr() + (packed_offset * 4) as u64) as *mut u32,
                )?;
            } else {
                self.module.llama_fusion.norm_pack::<false>(
                    &self.stream,
                    config,
                    input.len() / 2048,
                    self.eps,
                    input,
                    input,
                    weight,
                    input.cu_deviceptr() as *mut f32,
                    (packed.cu_deviceptr() + (packed_offset * 4) as u64) as *mut u32,
                )?;
            }
        }
        Ok(output)
    }
}
