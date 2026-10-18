//! Reuse prepared activations only inside one immutable-input projection group.
use super::*;

impl Engine {
    pub(super) fn mm_narrow_pair(
        &self,
        names: [&str; 2],
        input: &DeviceBuffer<f32>,
    ) -> Result<[Buffer; 2]> {
        let first = self.weight(names[0])?;
        let second = self.weight(names[1])?;
        if self.profile
            || first.kind != 30
            || second.kind != 30
            || first.shape.len() != 2
            || first.shape != second.shape
            || first.shape[0] == 0
            || first.shape[1] == 0
            || first.shape[1] > 96
            || !input.len().is_multiple_of(first.shape[0])
            || input.len() / first.shape[0] > 8
        {
            return self.mm_many(names, input);
        }
        let cols = first.shape[0];
        let rows = first.shape[1];
        let batch = input.len() / cols;
        let extent = rows.checked_mul(cols).and_then(|n| n.checked_mul(2));
        anyhow::ensure!(
            extent == Some(first.data.len()) && extent == Some(second.data.len()),
            "invalid paired BF16 weight extent"
        );
        let parts = cols.div_ceil(256);
        let count = rows * batch;
        let a = self.zeros(count)?;
        let b = self.zeros(count)?;
        let partial = self.zeros(2 * parts * count)?;
        // SAFETY: identical complete BF16 layouts, checked input width, disjoint
        // outputs, and the same partition/reduction order as independent mm.
        unsafe {
            self.module.bf16_pair.bf16_pair_partial(
                &self.stream,
                LaunchConfig::for_num_elems((2 * parts * count * 32) as u32),
                cols,
                rows,
                batch,
                parts,
                &first.data,
                &second.data,
                input,
                partial.cu_deviceptr() as *mut f32,
            )?;
            self.module.bf16_pair.bf16_pair_merge(
                &self.stream,
                LaunchConfig::for_num_elems((2 * count * 32) as u32),
                parts,
                &partial,
                a.cu_deviceptr() as *mut f32,
                b.cu_deviceptr() as *mut f32,
            )?;
        }
        Ok([a, b])
    }

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
