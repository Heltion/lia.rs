use super::*;
use anyhow::Context;

impl Engine {
    pub(super) fn feed_forward(&self, layer: usize, input: &DeviceBuffer<f32>) -> Result<Buffer> {
        let prefix = format!("blk.{layer}");
        // Decode GSQ projections read their input directly; routed register
        // experts have separate workspaces. Keep shared intermediates leased
        // until both branches join so the pool cannot alias concurrent work.
        let mut fork = if self.geometry.moe
            && input.len() == self.geometry.hidden
            && ["ffn_up", "ffn_gate", "ffn_down"].into_iter().all(|name| {
                self.weights
                    .get(&format!("{prefix}.{name}.weight"))
                    .is_some_and(|w| w.group.is_some())
            })
            && self.preallocated
            && !self.profile
            && self.experts.get(&layer).is_some_and(|e| e.is_register())
            && std::env::var("LIA_MOE_GRAPH_OVERLAP").as_deref() == Ok("1")
        {
            graph::branch::Frontier::current(&self.stream)?
        } else {
            None
        };
        let [up, gate] = self.mm_many(
            [
                &format!("{prefix}.ffn_up.weight"),
                &format!("{prefix}.ffn_gate.weight"),
            ],
            input,
        )?;
        let activated = self.activate_owned(up, &gate)?;
        let held_gate = if fork.is_some() {
            Some(gate)
        } else {
            drop(gate);
            None
        };
        let shared = self.mm(&format!("{prefix}.ffn_down.weight"), &activated)?;
        let mut shared_end = if let Some(start) = fork.as_mut() {
            let end = graph::branch::Frontier::current(&self.stream)?;
            start.restore(&self.stream)?;
            end
        } else {
            None
        };
        let held_activation = if fork.is_some() {
            Some(activated)
        } else {
            drop(activated);
            None
        };
        if !self.geometry.moe {
            return Ok(shared);
        }
        let batch = input.len() / self.geometry.hidden;
        let name = format!("{prefix}.ffn_gate_inp.weight");
        let weight = self.weight(&name)?;
        let fused = batch <= 8
            && weight.kind == 30
            && weight.group.is_none()
            && weight.shape == [self.geometry.hidden, 256]
            && self.geometry.hidden.is_multiple_of(128)
            && std::env::var("LIA_ROUTER_FUSED").as_deref() == Ok("1");
        let logits = if fused {
            self.zeros(batch * 256)?
        } else {
            self.mm(&name, input)?
        };
        let routing = self
            .routing
            .as_ref()
            .context("missing MoE routing workspace")?;
        anyhow::ensure!(
            logits.len() == batch * 256
                && routing.selected.len() >= batch * 8
                && routing.probabilities.len() >= batch * 8,
            "invalid MoE router extent"
        );
        // SAFETY: complete expert logits and eight bounded routing slots per
        // token. The following expert kernels consume this stream's results.
        unsafe {
            if fused {
                anyhow::ensure!(
                    weight.data.len() == self.geometry.hidden * 256 * 2
                        && routing.router_count.len() >= batch,
                    "invalid fused router buffers"
                );
                let prepared = self.module.router_fused.prepare_router_fused(
                    cuda_core::simt::LaunchConfig1D::new((batch * 32) as u32, 256, 0),
                )?;
                self.module.router_fused.router_fused(
                    &self.stream,
                    &prepared,
                    self.geometry.hidden,
                    batch,
                    &weight.data,
                    input,
                    logits.cu_deviceptr() as *mut f32,
                    routing.router_count.cu_deviceptr() as *mut u32,
                    routing.selected.cu_deviceptr() as *mut u32,
                    routing.probabilities.cu_deviceptr() as *mut f32,
                )?;
            } else {
                self.module.moe.route(
                    &self.stream,
                    LaunchConfig::for_num_elems((batch * 32) as u32),
                    &logits,
                    routing.selected.cu_deviceptr() as *mut u32,
                    routing.probabilities.cu_deviceptr() as *mut f32,
                )?;
            }
        }
        drop(logits);
        let mut activated = self.zeros(batch * 8 * 512)?;
        let mut routed = self.zeros(input.len())?;
        let experts = self
            .experts
            .get(&layer)
            .context("missing MoE expert block")?;
        if batch > 8 {
            let mut choices_out = self.zeros(batch * 8 * self.geometry.hidden)?;
            let packed = self.prefill_input.borrow();
            let packed_input = if std::env::var("LIA_MOE_PACK_INPUT").as_deref() == Ok("1") {
                Some(packed.as_ref().context("missing packed MoE workspace")?)
            } else {
                None
            };
            experts.forward_grouped(
                &self.module,
                &self.stream,
                &routing.grouped,
                input,
                packed_input,
                std::env::var("LIA_MOE_HALF_CODES").as_deref() == Ok("1"),
                &routing.selected,
                &routing.probabilities,
                &mut activated,
                &mut choices_out,
                &mut routed,
            )?;
        } else if experts.is_register() {
            experts.forward_register(
                &self.module,
                &self.stream,
                routing
                    .register
                    .as_ref()
                    .context("missing register MoE workspace")?,
                input,
                &routing.selected,
                &routing.probabilities,
                &mut activated,
                &mut routed,
            )?;
        } else {
            experts.forward(
                &self.module,
                &self.stream,
                input,
                &routing.selected,
                &routing.probabilities,
                &mut activated,
                &mut routed,
            )?;
        }
        drop(activated);
        let gate = self.mm(&format!("{prefix}.ffn_shared_gate.weight"), input)?;
        if let Some(end) = shared_end.as_mut() {
            end.join(&self.stream)?;
        }
        drop(held_activation);
        drop(held_gate);
        let result = self.zeros(input.len())?;
        // SAFETY: shared and routed arrays cover identical hidden rows; the
        // scalar shared gate has one element per token. Output is disjoint.
        unsafe {
            self.module.moe.combine(
                &self.stream,
                LaunchConfig::for_num_elems(input.len() as u32),
                self.geometry.hidden,
                &routed,
                &shared,
                &gate,
                result.cu_deviceptr() as *mut f32,
            )?;
        }
        Ok(result)
    }
}
