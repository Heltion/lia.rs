//! Checked activation output for paired 8-row gate/up weight layouts.
use super::*;

impl Weights {
    pub(crate) fn gateup_elements(&self, input: usize) -> Result<usize> {
        ensure!(
            self.row_interleave == 8 && self.rows.is_multiple_of(64),
            "AWQ gate/up activation requires paired 8-row interleaving"
        );
        Ok(self.output_elements(input)? / 2)
    }

    pub(crate) fn fused_gateup_eligible(&self, input: usize) -> bool {
        self.row_interleave == 8
            && self.rows.is_multiple_of(64)
            && self.columns > 0
            && input.is_multiple_of(self.columns)
            && input / self.columns > 8
            && !(49..64).contains(&(input / self.columns))
            && self.parts(input) == 1
            && self.fp16_prefill
            && self.group_size.is_multiple_of(64)
            && [
                input,
                self.rows,
                self.columns,
                self.group_size,
                self.packed.len(),
                self.zeros.len(),
                self.scales.len(),
            ]
            .into_iter()
            .all(|n| u32::try_from(n).is_ok())
    }

    pub(crate) fn matmul_gateup(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        ensure!(
            self.fused_gateup_eligible(input.len()),
            "unsupported fused AWQ gate/up shape or precision"
        );
        ensure!(
            output.len() == self.gateup_elements(input.len())?
                && u32::try_from(output.len()).is_ok(),
            "invalid fused AWQ gate/up output extent"
        );
        let batch = input.len() / self.columns;
        let (row_tile, tokens, threads) = launch_shape(batch);
        let blocks = (self.rows / row_tile)
            .checked_mul(batch.div_ceil(tokens))
            .context("AWQ gate/up grid overflow")?;
        ensure!(
            blocks <= (u32::MAX / threads) as usize,
            "AWQ gate/up thread index overflow"
        );
        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: paired row layout and whole K64 groups, bounded offsets;
        // each lane writes its two disjoint logical activation elements.
        unsafe {
            macro_rules! run {
                ($entry:ident) => {
                    module.awq_gateup.$entry(
                        stream,
                        config,
                        self.columns,
                        self.rows,
                        self.group_size,
                        batch,
                        &self.packed,
                        &self.zeros,
                        &self.scales,
                        input,
                        output.cu_deviceptr() as *mut f32,
                    )?;
                };
            }
            if batch >= 64 {
                run!(gateup_wide);
            } else if batch > 32 {
                run!(gateup_medium);
            } else {
                run!(gateup_narrow);
            }
        }
        Ok(())
    }
}

fn launch_shape(batch: usize) -> (usize, usize, u32) {
    if batch >= 64 {
        (64, 64, 256)
    } else if batch > 32 {
        (32, 48, 192)
    } else {
        (32, 32, 128)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project(
        weights: &Weights,
        module: &kernels::LoadedModule,
        stream: &Arc<CudaStream>,
        input: &DeviceBuffer<f32>,
    ) -> Result<DeviceBuffer<f32>> {
        let mut output = DeviceBuffer::zeroed(stream, weights.output_elements(input.len())?)?;
        let count = weights.workspace_elements(input.len())?;
        let mut workspace = if count == 0 {
            None
        } else {
            Some(DeviceBuffer::zeroed(stream, count)?)
        };
        weights.matmul(module, stream, input, workspace.as_mut(), &mut output)?;
        Ok(output)
    }

    #[test]
    #[ignore = "requires official MiniCPM-o AWQ checkpoint and CUDA device"]
    fn paired_activation_matches_separate_projections() -> Result<()> {
        let directory =
            std::env::var("LIA_MINICPMO45_MODEL").context("set LIA_MINICPMO45_MODEL")?;
        let archive = crate::safetensors::Archive::open(directory)?;
        let matrices = [
            Matrix::load(&archive, "llm.model.layers.0.mlp.gate_proj", 128)?,
            Matrix::load(&archive, "llm.model.layers.0.mlp.up_proj", 128)?,
        ];
        let paired = Matrix::interleave_pair(&matrices, 8)?;
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        let mut gate = Weights::upload(&stream, &matrices[0])?;
        let mut up = Weights::upload(&stream, &matrices[1])?;
        let mut joined = Weights::upload(&stream, &paired)?;
        ensure!(
            gate.gateup_elements(gate.columns).is_err(),
            "ordinary layout accepted"
        );
        for fp16 in [true, false] {
            for weight in [&mut gate, &mut up, &mut joined] {
                weight.tensor_prefill = true;
                weight.fp16_prefill = fp16;
            }
            for batch in [1, 9, 32, 34, 48, 49, 63, 64, 65, 129, 320, 512] {
                let values: Vec<f32> = (0..gate.columns * batch)
                    .map(|i| ((i * 17 % 257) as f32 - 128.0) * 0.0137)
                    .collect();
                let input = DeviceBuffer::from_host(&stream, &values)?;
                let dg = project(&gate, &module, &stream, &input)?;
                let du = project(&up, &module, &stream, &input)?;
                let count = gate.rows * batch;
                let config = LaunchConfig::for_num_elems(u32::try_from(count)?);
                let mut expected = DeviceBuffer::zeroed(&stream, count)?;
                let mut actual = DeviceBuffer::zeroed(&stream, count)?;
                // SAFETY: all activation buffers contain exactly count elements.
                unsafe {
                    module
                        .basic
                        .silu_mul(&stream, config, &du, &dg, &mut expected)?;
                }
                if joined.fused_gateup_eligible(input.len()) {
                    joined.matmul_gateup(&module, &stream, &input, &mut actual)?;
                    let mut guarded =
                        DeviceBuffer::from_host(&stream, &vec![12345.0f32; count + 17])?;
                    ensure!(
                        joined
                            .matmul_gateup(&module, &stream, &input, &mut guarded)
                            .is_err(),
                        "checked launch accepted excess output"
                    );
                    let (rt, nt, threads) = launch_shape(batch);
                    let config = LaunchConfig {
                        grid_dim: (((joined.rows / rt) * batch.div_ceil(nt)) as u32, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    // SAFETY: same validated launch with additional trailing guards.
                    unsafe {
                        macro_rules! raw {
                            ($entry:ident) => {
                                module.awq_gateup.$entry(
                                    &stream,
                                    config,
                                    joined.columns,
                                    joined.rows,
                                    joined.group_size,
                                    batch,
                                    &joined.packed,
                                    &joined.zeros,
                                    &joined.scales,
                                    &input,
                                    guarded.cu_deviceptr() as *mut f32,
                                )?;
                            };
                        }
                        if batch >= 64 {
                            raw!(gateup_wide);
                        } else if batch > 32 {
                            raw!(gateup_medium);
                        } else {
                            raw!(gateup_narrow);
                        }
                    }
                    let guard = guarded.to_host_vec(&stream)?;
                    ensure!(
                        guard[count..].iter().all(|v| *v == 12345.0),
                        "output guard overwritten"
                    );
                    ensure!(
                        guard[..count]
                            .iter()
                            .zip(actual.to_host_vec(&stream)?)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "raw launch mismatch"
                    );
                } else {
                    let projection = project(&joined, &module, &stream, &input)?;
                    // SAFETY: paired row layout and exact logical activation extent.
                    unsafe {
                        module.basic.silu_interleaved(
                            &stream,
                            config,
                            gate.rows,
                            &projection,
                            &mut actual,
                        )?;
                    }
                }
                for (i, (a, b)) in actual
                    .to_host_vec(&stream)?
                    .into_iter()
                    .zip(expected.to_host_vec(&stream)?)
                    .enumerate()
                {
                    ensure!(
                        a.to_bits() == b.to_bits(),
                        "batch {batch} fp16={fp16} element {i}: actual {a} expected {b}"
                    );
                }
                println!("paired activation batch={batch} fp16={fp16}: bit-exact, guards intact");
            }
        }
        Ok(())
    }
}
