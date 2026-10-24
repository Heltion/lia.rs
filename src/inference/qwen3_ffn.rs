//! Qwen3 AWQ feed-forward layouts select a fused or exact fallback activation.
use super::*;

impl Engine {
    pub(super) fn qwen3_awq_feed_forward(
        &self,
        prefix: &str,
        input: &DeviceBuffer<f32>,
    ) -> Result<Option<Buffer>> {
        let interleaved = format!("{prefix}.ffn_gateup_interleaved.weight");
        let joined_name = format!("{prefix}.ffn_gateup.weight");
        let batch = input.len() / self.geometry.hidden;
        let activated = if let Some(weight) = self.awq_weights.get(&interleaved) {
            let mut activated = self.zeros(weight.gateup_elements(input.len())?)?;
            if weight.fused_gateup_eligible(input.len()) {
                weight.matmul_gateup(&self.module, &self.stream, input, &mut activated)?;
            } else {
                let joined = self.mm(&interleaved, input)?;
                anyhow::ensure!(
                    activated.len() == batch * 12288,
                    "invalid interleaved Qwen3 FFN shape"
                );
                // SAFETY: marker8 and complete pairs validated by gateup_elements.
                unsafe {
                    self.module.basic.silu_interleaved(
                        &self.stream,
                        LaunchConfig::for_num_elems(u32::try_from(activated.len())?),
                        12288,
                        &joined,
                        &mut activated,
                    )?;
                }
            }
            activated
        } else if self.awq_weights.contains_key(&joined_name) {
            let joined = self.mm(&joined_name, input)?;
            anyhow::ensure!(
                joined.len() == batch * 24576,
                "invalid joined Qwen3 FFN shape"
            );
            let mut activated = self.zeros(batch * 12288)?;
            // SAFETY: complete gate/up halves and exact bounded activation output.
            unsafe {
                self.module.basic.silu_joined(
                    &self.stream,
                    LaunchConfig::for_num_elems(u32::try_from(activated.len())?),
                    12288,
                    &joined,
                    &mut activated,
                )?;
            }
            activated
        } else {
            return Ok(None);
        };
        self.mm(&format!("{prefix}.ffn_down.weight"), &activated)
            .map(Some)
    }
}
