//! Checked tensor launches; small quantization groups retain the generic path.
use super::*;

impl Weights {
    pub(super) fn matmul_prefill(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: &DeviceBuffer<f32>,
        workspace: Option<&mut DeviceBuffer<f32>>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let batch = input.len() / self.columns;
        let parts = self.parts(input.len());
        ensure!(
            workspace.as_ref().map_or(0, |w| w.len()) == self.workspace_elements(input.len())?,
            "invalid AWQ prefill partial workspace shape"
        );
        if parts > 1 {
            let partial = workspace.context("missing AWQ prefill partial workspace")?;
            let medium = batch > 32;
            let tokens = if medium { 48 } else { 32 };
            let threads = if medium { 192 } else { 128 };
            let blocks = self
                .rows
                .div_ceil(32)
                .checked_mul(batch.div_ceil(tokens))
                .and_then(|n| n.checked_mul(parts))
                .context("AWQ split grid overflow")?;
            ensure!(
                blocks <= (u32::MAX / threads) as usize && u32::try_from(partial.len()).is_ok(),
                "AWQ split offsets overflow"
            );
            let config = LaunchConfig {
                grid_dim: (blocks as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: complete CTAs, group-aligned K partitions, checked offsets;
            // each partial output is written once before the serial F32 merge.
            unsafe {
                macro_rules! split {
                    ($entry:ident) => {
                        module.awq_split.$entry(
                            stream,
                            config,
                            self.columns,
                            self.rows,
                            self.group_size,
                            batch,
                            parts,
                            &self.packed,
                            &self.zeros,
                            &self.scales,
                            input,
                            partial.cu_deviceptr() as *mut f32,
                        )?;
                    };
                }
                if medium {
                    split!(medium);
                } else {
                    split!(narrow);
                }
                module.awq.awq_merge(
                    stream,
                    LaunchConfig::for_num_elems(u32::try_from(output.len())?),
                    self.rows,
                    batch,
                    parts,
                    partial,
                    output.cu_deviceptr() as *mut f32,
                )?;
            }
            return Ok(());
        }
        let mut config = LaunchConfig {
            grid_dim: (
                u32::try_from(self.rows.div_ceil(32) * batch.div_ceil(32))
                    .context("AWQ tensor grid overflow")?,
                1,
                1,
            ),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: parent validates complete input/output rows and packed words.
        // Kernels use complete four-, six- or eight-warp CTAs, guarded tails, and static
        // shared tiles. K64 stays within groups; its element offsets and flattened
        // thread indices must fit u32, otherwise the generic path is retained.
        macro_rules! launch {
            ($field:ident, $entry:ident $(, $residual:literal)?) => {{
                unsafe { module.$field.$entry $(::<$residual>)? (
                    stream, config, self.columns, self.rows, self.group_size, batch,
                    &self.packed, &self.zeros, &self.scales, input, output.cu_deviceptr() as *mut f32,
                )?; }
            }};
        }
        let narrow = [
            self.columns,
            self.rows,
            self.group_size,
            self.packed.len(),
            self.zeros.len(),
            self.scales.len(),
            input.len(),
            output.len(),
        ]
        .into_iter()
        .all(|n| u32::try_from(n).is_ok())
            && config.grid_dim.0 <= u32::MAX / 128;
        if self.fp16_prefill && self.group_size.is_multiple_of(64) && narrow {
            if (33..=48).contains(&batch) {
                config.grid_dim.0 = u32::try_from(self.rows.div_ceil(32) * batch.div_ceil(48))
                    .context("AWQ medium tensor grid overflow")?;
                config.block_dim.0 = 192;
                ensure!(
                    config.grid_dim.0 <= u32::MAX / 192,
                    "AWQ medium thread index overflow"
                );
                launch!(awq_medium, awq_fp16_medium);
            } else if batch >= 64 && self.rows >= 64 {
                config.grid_dim.0 = u32::try_from(self.rows.div_ceil(64) * batch.div_ceil(64))
                    .context("AWQ square tensor grid overflow")?;
                config.block_dim.0 = 256;
                ensure!(
                    config.grid_dim.0 <= u32::MAX / 256,
                    "AWQ square thread index overflow"
                );
                launch!(awq_tile, awq_fp16_square);
            } else if batch >= 64 {
                config.grid_dim.0 = u32::try_from(self.rows.div_ceil(32) * batch.div_ceil(64))
                    .context("AWQ wide tensor grid overflow")?;
                launch!(awq_tile, awq_fp16_wide);
            } else {
                launch!(awq_fp16, awq_fp16);
            }
        } else if self.fp16_prefill {
            launch!(awq_matrix, awq_matrix, false);
        } else {
            launch!(awq_matrix, awq_matrix, true);
        }
        Ok(())
    }
}
