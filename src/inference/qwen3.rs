//! Standard Qwen3 attention; hybrid Qwen and MiniCPM5 retain their own paths.
use super::*;

impl Engine {
    pub(super) fn qwen3_attention(
        &self,
        layer: usize,
        input: &DeviceBuffer<f32>,
    ) -> Result<Buffer> {
        let batch = input.len() / self.geometry.hidden;
        let prefix = format!("blk.{layer}");
        let combined = format!("{prefix}.attn_qkv.weight");
        let [q, k, v] = if self.awq_weights.contains_key(&combined) {
            let joined = self.mm(&combined, input)?;
            anyhow::ensure!(
                joined.len() == batch * 6144,
                "invalid fused Qwen3 projection extent"
            );
            let q = self.zeros(batch * 4096)?;
            let k = self.zeros(batch * 1024)?;
            let v = self.zeros(batch * 1024)?;
            // SAFETY: the joined matrix has fixed Q/K/V segment widths, and
            // each checked complete token is split into disjoint output slots.
            unsafe {
                self.module.dense_attention.split_qkv(
                    &self.stream,
                    LaunchConfig::for_num_elems(u32::try_from(joined.len())?),
                    &joined,
                    q.cu_deviceptr() as *mut f32,
                    k.cu_deviceptr() as *mut f32,
                    v.cu_deviceptr() as *mut f32,
                )?;
            }
            [q, k, v]
        } else {
            self.mm_many(
                [
                    &format!("{prefix}.attn_q.weight"),
                    &format!("{prefix}.attn_k.weight"),
                    &format!("{prefix}.attn_v.weight"),
                ],
                input,
            )?
        };
        anyhow::ensure!(
            q.len() == batch * 4096
                && k.len() == batch * 1024
                && v.len() == k.len()
                && self.position + batch <= self.capacity,
            "invalid Qwen3 projection or cache extent"
        );
        let rotated_q = self.qwen3_prepare(&prefix, layer, &q, &k, &v)?;
        let State::Attention { k: kc, v: vc } = &self.states[layer] else {
            anyhow::bail!("Qwen3 requires a full attention cache at every layer")
        };
        let attended = self.zeros(q.len())?;
        let span = if batch == 1 && self.position >= self.attention_decode_span {
            self.attention_decode_span
        } else {
            0
        };
        let parts = if span == 0 {
            1
        } else {
            (self.position + 1).div_ceil(span)
        };
        let partial = if span == 0 {
            None
        } else {
            Some(self.zeros(32 * parts * 130)?)
        };
        let destination = partial.as_ref().unwrap_or(&attended);
        // SAFETY: validated 32/8-head complete 128-value rows; RMS weights
        // matched their widths. Causal cache accesses end at position + batch.
        unsafe {
            if batch >= 16 {
                if self.qwen3_register_p {
                    self.module.qwen3_flash.qwen3_flash_register(
                        &self.stream,
                        LaunchConfig {
                            grid_dim: ((batch * 4).div_ceil(64) as u32 * 8, 1, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        batch,
                        self.position,
                        &rotated_q,
                        kc,
                        vc,
                        attended.cu_deviceptr() as *mut f32,
                    )?;
                } else if self.qwen3_tensor_pv {
                    self.module.qwen3_flash.qwen3_flash_residual(
                        &self.stream,
                        LaunchConfig {
                            grid_dim: ((batch * 4).div_ceil(64) as u32 * 8, 1, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        batch,
                        self.position,
                        &rotated_q,
                        kc,
                        vc,
                        attended.cu_deviceptr() as *mut f32,
                    )?;
                } else {
                    self.module.qwen3_flash.qwen3_flash(
                        &self.stream,
                        LaunchConfig {
                            grid_dim: ((batch * 4).div_ceil(64) as u32 * 8, 1, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        batch,
                        self.position,
                        &rotated_q,
                        kc,
                        vc,
                        attended.cu_deviceptr() as *mut f32,
                    )?;
                }
            } else {
                self.module.dense_attention.dense_attention128(
                    &self.stream,
                    LaunchConfig::for_num_elems((batch * 32 * parts * 32) as u32),
                    batch,
                    32,
                    8,
                    self.position,
                    parts,
                    span,
                    &rotated_q,
                    kc,
                    vc,
                    destination.cu_deviceptr() as *mut f32,
                )?;
                if let Some(partial) = &partial {
                    self.module.dense_attention.dense_attention_merge(
                        &self.stream,
                        LaunchConfig::for_num_elems(32 * 32),
                        32,
                        parts,
                        partial,
                        attended.cu_deviceptr() as *mut f32,
                    )?;
                }
            }
        }
        self.mm(&format!("{prefix}.attn_output.weight"), &attended)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::ensure;

    #[test]
    #[ignore = "requires official MiniCPM-o AWQ checkpoint and CUDA"]
    fn concatenated_qkv_matches_independent_projections() -> Result<()> {
        let directory = std::env::var("LIA_MINICPMO45_MODEL")?;
        let archive = crate::safetensors::Archive::open(directory)?;
        let matrices = ["q_proj", "k_proj", "v_proj"]
            .map(|name| {
                crate::quant::awq::Matrix::load(
                    &archive,
                    &format!("llm.model.layers.0.self_attn.{name}"),
                    128,
                )
            })
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        let joined = crate::gpu::awq::Weights::upload(
            &stream,
            &crate::quant::awq::Matrix::concatenate(&matrices)?,
        )?;
        let independent = matrices
            .iter()
            .map(|m| crate::gpu::awq::Weights::upload(&stream, m))
            .collect::<Result<Vec<_>>>()?;
        let apply = |weight: &crate::gpu::awq::Weights,
                     input: &DeviceBuffer<f32>|
         -> Result<DeviceBuffer<f32>> {
            let mut output = DeviceBuffer::zeroed(&stream, weight.output_elements(input.len())?)?;
            let n = weight.workspace_elements(input.len())?;
            let mut partial = if n == 0 {
                None
            } else {
                Some(DeviceBuffer::zeroed(&stream, n)?)
            };
            weight.matmul(&module, &stream, input, partial.as_mut(), &mut output)?;
            Ok(output)
        };
        for batch in [1usize, 34, 65, 512] {
            let input: Vec<f32> = (0..batch * 4096)
                .map(|i| ((i * 17 % 257) as f32 - 128.) * 0.0137)
                .collect();
            let input = DeviceBuffer::from_host(&stream, &input)?;
            let combined = apply(&joined, &input)?;
            let guard = 12345.0f32;
            let q = DeviceBuffer::from_host(&stream, &vec![guard; batch * 4096 + 17])?;
            let k = DeviceBuffer::from_host(&stream, &vec![guard; batch * 1024 + 17])?;
            let v = DeviceBuffer::from_host(&stream, &vec![guard; batch * 1024 + 17])?;
            // SAFETY: checked matrix dimensions are fixed 6144-wide Q/K/V,
            // each output includes its complete shape and trailing guards.
            unsafe {
                module.dense_attention.split_qkv(
                    &stream,
                    LaunchConfig::for_num_elems(u32::try_from(combined.len())?),
                    &combined,
                    q.cu_deviceptr() as *mut f32,
                    k.cu_deviceptr() as *mut f32,
                    v.cu_deviceptr() as *mut f32,
                )?;
            }
            let mut maximum = 0.0f32;
            for (index, output) in [q, k, v].into_iter().enumerate() {
                let actual = output.to_host_vec(&stream)?;
                let expected = apply(&independent[index], &input)?.to_host_vec(&stream)?;
                ensure!(
                    actual[expected.len()..].iter().all(|v| *v == guard),
                    "QKV split guard overwritten"
                );
                for (a, b) in actual.iter().zip(expected) {
                    let error = (*a - b).abs();
                    maximum = maximum.max(error);
                    ensure!(
                        a.is_finite() && error <= 2e-4 * (1.0 + b.abs()),
                        "QKV projection {index} batch {batch}: {error}"
                    );
                }
            }
            println!("combined QKV batch={batch}: max_error={maximum}, guards intact");
        }
        Ok(())
    }
}
