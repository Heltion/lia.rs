//! Reuse prepared activations only inside one immutable-input projection group.
use super::*;

impl Engine {
    pub(super) fn mm_many<const N: usize>(
        &self,
        names: [&str; N],
        input: &DeviceBuffer<f32>,
    ) -> Result<[Buffer; N]> {
        let first = names
            .first()
            .ok_or_else(|| anyhow!("empty projection group"))?;
        let shape = &self.weight(first)?.shape;
        anyhow::ensure!(shape.len() == 2 && shape[0] > 0, "invalid projection shape");
        let cols = shape[0];
        let batch = input.len() / cols;
        let (mut ready, mut fp8_ready) = (false, false);
        let mut outputs: [Option<Buffer>; N] = std::array::from_fn(|_| None);
        for (output, name) in outputs.iter_mut().zip(names) {
            let weight = self.weight(name)?;
            anyhow::ensure!(
                weight.shape.len() == 2 && weight.shape[0] == cols,
                "projection input width mismatch: {name}"
            );
            let fp8 = batch >= crate::gpu::WIDE_MIN_BATCH && weight.fp8_scale.is_some();
            *output = Some(self.mm_prepared(name, input, if fp8 { fp8_ready } else { ready })?);
            // Reuse is tracked separately for FP8 and the existing representation.
            // Short F32/BF16 projections leave integer activations untouched.
            if fp8 {
                fp8_ready = true;
            } else {
                ready |= batch > 8 || (self.integer && weight.kind != 0 && weight.kind != 30);
            }
        }
        Ok(outputs.map(|output| output.expect("all projections initialized")))
    }
}
