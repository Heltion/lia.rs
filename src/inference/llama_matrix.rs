use super::*;
use anyhow::Context;
impl Engine {
    pub(super) fn llama_qkv_prefill(
        &self,
        layer: usize,
        input: &DeviceBuffer<f32>,
        prepared: bool,
    ) -> Result<Option<Buffer>> {
        let batch = input.len() / 2048;
        let Some(weights) = self
            .llama_weight_cache
            .get(&format!("blk.{layer}.attn_qkv.weight"))
        else {
            return Ok(None);
        };
        if batch <= 8 || self.paging.is_none() {
            return Ok(None);
        }
        let output = self.zeros(batch * 2560)?;
        let mut workspace = self.prefill_input.borrow_mut();
        if workspace
            .as_ref()
            .is_none_or(|buffer| buffer.len() < input.len() / 2)
        {
            anyhow::ensure!(
                !self.preallocated,
                "combined projection input workspace exhausted"
            );
            *workspace = Some(DeviceBuffer::zeroed(&self.stream, input.len() / 2)?);
        }
        let packed = workspace
            .as_ref()
            .context("missing combined projection workspace")?;
        // SAFETY: combined F16 rows preserve Q/K/V order; the packed input is
        // either this batch's normalized input or freshly packed by the launcher.
        unsafe {
            crate::gpu::launch_prefill_matrix(
                &self.module,
                &self.stream,
                1,
                2048,
                2560,
                batch,
                weights,
                input,
                prepared,
                packed,
                output.cu_deviceptr() as *mut f32,
                self.prefill_tile,
            )?;
        }
        Ok(Some(output))
    }

    pub(super) fn llama_bf16_matrix(
        &self,
        weight: &Weight,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
    ) -> Result<()> {
        let columns = weight.shape[0];
        let rows = weight.shape[1];
        let batch = input.len() / columns;
        anyhow::ensure!(
            columns.is_multiple_of(512) && (1..=8).contains(&batch),
            "invalid vector BF16 geometry"
        );
        anyhow::ensure!(
            weight.data.len() == columns * rows * 2 && output.len() == rows * batch,
            "invalid vector BF16 extent"
        );
        anyhow::ensure!(
            weight.data.cu_deviceptr().is_multiple_of(16)
                && input.cu_deviceptr().is_multiple_of(16),
            "unaligned vector BF16 operands"
        );
        let config = LaunchConfig {
            grid_dim: (rows.div_ceil(4) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: complete aligned eight-value loads, validated extents, and
        // a complete warp per row. Compile-time lanes cover each runtime batch.
        unsafe {
            match batch {
                1 => self.module.llama_bf16.llama_bf16_vector::<1>(
                    &self.stream,
                    config,
                    columns,
                    rows,
                    batch,
                    &weight.data,
                    input,
                    output.cu_deviceptr() as *mut f32,
                )?,
                2 => self.module.llama_bf16.llama_bf16_vector::<2>(
                    &self.stream,
                    config,
                    columns,
                    rows,
                    batch,
                    &weight.data,
                    input,
                    output.cu_deviceptr() as *mut f32,
                )?,
                3..=4 => self.module.llama_bf16.llama_bf16_vector::<4>(
                    &self.stream,
                    config,
                    columns,
                    rows,
                    batch,
                    &weight.data,
                    input,
                    output.cu_deviceptr() as *mut f32,
                )?,
                _ => self.module.llama_bf16.llama_bf16_vector::<8>(
                    &self.stream,
                    config,
                    columns,
                    rows,
                    batch,
                    &weight.data,
                    input,
                    output.cu_deviceptr() as *mut f32,
                )?,
            }
        }
        Ok(())
    }
}
