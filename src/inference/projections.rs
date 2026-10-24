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
        if names
            .iter()
            .any(|name| self.awq_weights.contains_key(*name))
        {
            let mut outputs: [Option<Buffer>; N] = std::array::from_fn(|_| None);
            for (out, name) in outputs.iter_mut().zip(names) {
                *out = Some(self.mm(name, input)?);
            }
            return Ok(outputs.map(Option::unwrap));
        }
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
                let full_precision = weight.group.is_none()
                    && weight.kind == 30
                    && (weight.shape[1] <= 96
                        || (self.geometry.moe
                            && weight.shape[1] == 256
                            && cols.is_multiple_of(128)));
                ready |= (batch > 8 && !full_precision)
                    || (self.integer && weight.kind != 0 && weight.kind != 30);
            }
        }
        Ok(outputs.map(|output| output.expect("all projections initialized")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context, ensure};

    #[test]
    #[ignore = "requires LIA_TEST_MODEL with a local MiniCPM checkpoint"]
    fn mixed_precision_groups_prepare_tensor_input() -> Result<()> {
        let model = std::env::var("LIA_TEST_MODEL").context("set LIA_TEST_MODEL")?;
        let engine = Engine::load_huggingface(&model, 1024, false)?;
        ensure!(engine.geometry.hidden == 2048, "unsupported test geometry");
        for (case, first) in ["blk.0.ssm_alpha.weight", "blk.0.ffn_gate_inp.weight"]
            .into_iter()
            .enumerate()
        {
            let input: Vec<f32> = (0..17 * 2048)
                .map(|i| (i % 113) as f32 * 0.0017 + case as f32 * 0.031)
                .collect();
            let input = DeviceBuffer::from_host(&engine.stream, &input)?;
            let names = [first, "blk.0.attn_qkv.weight"];
            let actual = engine.mm_many(names, &input)?;
            for (name, actual) in names.into_iter().zip(actual) {
                let expected = engine.mm(name, &input)?;
                ensure!(
                    actual.to_host_vec(&engine.stream)? == expected.to_host_vec(&engine.stream)?,
                    "mixed projection group reused unprepared input for {name}"
                );
            }
        }
        Ok(())
    }
}
