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
        let mut ready = false;
        let mut outputs: [Option<Buffer>; N] = std::array::from_fn(|_| None);
        for (output, name) in outputs.iter_mut().zip(names) {
            let weight = self.weight(name)?;
            anyhow::ensure!(
                weight.shape.len() == 2 && weight.shape[0] == cols,
                "projection input width mismatch: {name}"
            );
            *output = Some(self.mm_prepared(name, input, ready)?);
            // Large batches always pack f16 activations. Short batches prepare
            // integer inputs only for quantized weights; F32/BF16 leave them intact.
            ready |= batch > 8 || (self.integer && weight.kind != 0 && weight.kind != 30);
        }
        Ok(outputs.map(|output| output.expect("all projections initialized")))
    }
}
