use super::*;
use anyhow::Context;

impl Engine {
    pub(super) fn feed_forward(&self, layer: usize, input: &DeviceBuffer<f32>) -> Result<Buffer> {
        let prefix = format!("blk.{layer}");
        let [up, gate] = self.mm_many(
            [
                &format!("{prefix}.ffn_up.weight"),
                &format!("{prefix}.ffn_gate.weight"),
            ],
            input,
        )?;
        let activated = self.activate_owned(up, &gate)?;
        drop(gate);
        let shared = self.mm(&format!("{prefix}.ffn_down.weight"), &activated)?;
        drop(activated);
        if !self.geometry.moe {
            return Ok(shared);
        }
        let batch = input.len() / self.geometry.hidden;
        let logits = self.mm(&format!("{prefix}.ffn_gate_inp.weight"), input)?;
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
            self.module.moe.route(
                &self.stream,
                LaunchConfig::for_num_elems((batch * 32) as u32),
                &logits,
                routing.selected.cu_deviceptr() as *mut u32,
                routing.probabilities.cu_deviceptr() as *mut f32,
            )?;
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
            experts.forward_grouped(
                &self.module,
                &self.stream,
                &routing.grouped,
                input,
                &routing.selected,
                &routing.probabilities,
                &mut activated,
                &mut choices_out,
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
