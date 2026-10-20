//! Caller-owned split-K workspace and register-MMA expert execution.
use super::*;

pub(crate) struct RegisterWork {
    staged: bool,
    partial: DeviceBuffer<f32>,
}

impl RegisterWork {
    pub fn new(
        stream: &std::sync::Arc<CudaStream>,
        batch: usize,
        hidden: usize,
        intermediate: usize,
        staged: bool,
    ) -> Result<Self> {
        ensure!(
            batch > 0 && hidden > 0 && intermediate > 0,
            "invalid register MoE capacity"
        );
        let gate = batch * 8 * hidden.div_ceil(128) * intermediate * 2;
        let down = batch * 8 * intermediate.div_ceil(128) * hidden;
        Ok(Self {
            staged,
            partial: DeviceBuffer::zeroed(stream, gate.max(down))?,
        })
    }
}

impl Experts {
    #[allow(clippy::too_many_arguments)]
    pub fn forward_register(
        &self,
        module: &kernels::LoadedModule,
        stream: &std::sync::Arc<CudaStream>,
        work: &RegisterWork,
        input: &DeviceBuffer<f32>,
        selected: &DeviceBuffer<u32>,
        probabilities: &DeviceBuffer<f32>,
        activated: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if work.staged {
            self.forward_register_impl::<true>(
                module,
                stream,
                work,
                input,
                selected,
                probabilities,
                activated,
                output,
            )
        } else {
            self.forward_register_impl::<false>(
                module,
                stream,
                work,
                input,
                selected,
                probabilities,
                activated,
                output,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_register_impl<const STAGED: bool>(
        &self,
        module: &kernels::LoadedModule,
        stream: &std::sync::Arc<CudaStream>,
        work: &RegisterWork,
        input: &DeviceBuffer<f32>,
        selected: &DeviceBuffer<u32>,
        probabilities: &DeviceBuffer<f32>,
        activated: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        ensure!(
            !input.is_empty() && input.len().is_multiple_of(self.hidden),
            "invalid register MoE input"
        );
        let batch = input.len() / self.hidden;
        let gate_parts = self.hidden.div_ceil(128);
        let down_parts = self.intermediate.div_ceil(128);
        ensure!(
            self.gate.is_register()
                && self.up.is_register()
                && self.down.is_register()
                && self.hidden.is_multiple_of(128)
                && self.intermediate.is_multiple_of(128)
                && !self.gate.is_offloaded()
                && !self.up.is_offloaded()
                && !self.down.is_offloaded()
                && selected.len() >= batch * 8
                && probabilities.len() >= batch * 8
                && activated.len() == batch * 8 * self.intermediate
                && output.len() == batch * self.hidden
                && work.partial.len() >= batch * 8 * gate_parts * self.intermediate * 2
                && work.partial.len() >= batch * 8 * down_parts * self.hidden,
            "invalid register MoE storage or workspace extent"
        );
        let config = |rows: usize, parts: usize| LaunchConfig {
            grid_dim: ((batch * 8 * rows / 64 * parts) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the caller supplies repacked normal-scale resident matrices.
        // Validated buffer extents cover every complete warp tile and split;
        // all stages reuse one workspace in stream order, without allocation.
        unsafe {
            module.moe_selected.selected_projection::<true, STAGED>(
                stream,
                config(self.intermediate, gate_parts),
                self.intermediate,
                self.hidden,
                gate_parts,
                batch * 8,
                selected,
                input,
                &self.gate.data,
                &self.gate.scales,
                &self.up.data,
                &self.up.scales,
                work.partial.cu_deviceptr() as *mut f32,
            )?;
            module.moe_selected.selected_gate_merge(
                stream,
                LaunchConfig::for_num_elems(activated.len() as u32),
                self.intermediate,
                gate_parts,
                batch * 8,
                &work.partial,
                activated.cu_deviceptr() as *mut f32,
            )?;
            module.moe_selected.selected_projection::<false, STAGED>(
                stream,
                config(self.hidden, down_parts),
                self.hidden,
                self.intermediate,
                down_parts,
                batch * 8,
                selected,
                activated,
                &self.down.data,
                &self.down.scales,
                &self.down.data,
                &self.down.scales,
                work.partial.cu_deviceptr() as *mut f32,
            )?;
            module.moe_selected.selected_down_merge(
                stream,
                LaunchConfig::for_num_elems(output.len() as u32),
                self.hidden,
                down_parts,
                batch,
                &work.partial,
                probabilities,
                output.cu_deviceptr() as *mut f32,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_register_moe_matches_independent_expert_network() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        for (hidden, intermediate, batch) in [
            (128, 128, 2),
            (1024, 128, 1),
            (128, 1024, 2),
            (128, 128, 9),
            (128, 128, 33),
            (128, 128, 129),
            (128, 128, 257),
            (2048, 512, 1),
        ] {
            let mut bit_reference: Option<(Vec<u32>, Vec<u32>)> = None;
            for staged in [false, true] {
                let matrix = |rows: usize, columns: usize, offset: usize| Matrix {
                    rows: rows * 256,
                    columns,
                    bits: 2,
                    group_size: 128,
                    packed: (0usize..rows * 256 * columns / 4)
                        .map(|i| ((i * 73 + i / 97 + offset) ^ (i >> 5)) as u8)
                        .collect(),
                    scales: (0..rows * 256 * columns / 128)
                        .map(|i| {
                            f32::from_bits(
                                (((115 << 7) | (i * 17 % 128) | ((i + offset) % 2 << 15)) as u32)
                                    << 16,
                            )
                        })
                        .collect(),
                };
                let gate = matrix(intermediate, hidden, 17);
                let up = matrix(intermediate, hidden, 59);
                let down = matrix(hidden, intermediate, 97);
                let ids = (0..batch * 8)
                    .map(|i| ((i * 73 + i / 8 * 51) % 256) as u32)
                    .collect::<Vec<_>>();
                let probabilities = (0..batch * 8)
                    .map(|i| (i % 8 + 1) as f32 / 36.0)
                    .collect::<Vec<_>>();
                let input = (0..batch * hidden)
                    .map(|i| ((i * 19 % 127) as i32 - 63) as f32 / 33.0)
                    .collect::<Vec<_>>();
                let dot = |matrix: &Matrix, row: usize, input: &[f32]| -> f32 {
                    (0..matrix.columns)
                        .map(|c| {
                            let w = matrix.value(row, c).unwrap();
                            let x = input[c];
                            if batch > 8 {
                                ((w as f16) as f64) * ((x as f16) as f64)
                            } else {
                                w as f64 * x as f64
                            }
                        })
                        .sum::<f64>() as f32
                };
                let mut expected_act = vec![0.0; batch * 8 * intermediate];
                let mut expected = vec![0.0; batch * hidden];
                for choice in 0..batch * 8 {
                    for row in 0..intermediate {
                        let expert_row = ids[choice] as usize * intermediate + row;
                        let g = dot(&gate, expert_row, &input[choice / 8 * hidden..][..hidden]);
                        let u = dot(&up, expert_row, &input[choice / 8 * hidden..][..hidden]);
                        expected_act[choice * intermediate + row] = g / (1.0 + (-g).exp()) * u;
                    }
                }
                for token in 0..batch {
                    for row in 0..hidden {
                        for slot in 0..8 {
                            let choice = token * 8 + slot;
                            expected[token * hidden + row] += probabilities[choice]
                                * dot(
                                    &down,
                                    ids[choice] as usize * hidden + row,
                                    &expected_act[choice * intermediate..][..intermediate],
                                );
                        }
                    }
                }
                let mut experts =
                    Experts::upload_offloaded(&stream, hidden, intermediate, gate, up, down, 0)?;
                assert!(experts.prepare_register(&module, &stream)?);
                assert!(!experts.prepare_register(&module, &stream)?);
                let work = RegisterWork::new(&stream, batch, hidden, intermediate, staged)?;
                let x = DeviceBuffer::from_host(&stream, &input)?;
                let selected = DeviceBuffer::from_host(&stream, &ids)?;
                let probs = DeviceBuffer::from_host(&stream, &probabilities)?;
                let mut activated = DeviceBuffer::zeroed(&stream, expected_act.len())?;
                let mut output = DeviceBuffer::zeroed(&stream, expected.len())?;
                if batch > 8 {
                    let grouped = Grouped::new(&stream, batch)?;
                    let mut choices = DeviceBuffer::zeroed(&stream, batch * 8 * hidden)?;
                    experts.forward_grouped(
                        &module,
                        &stream,
                        &grouped,
                        &x,
                        None,
                        false,
                        &selected,
                        &probs,
                        &mut activated,
                        &mut choices,
                        &mut output,
                    )?;
                } else {
                    experts.forward_register(
                        &module,
                        &stream,
                        &work,
                        &x,
                        &selected,
                        &probs,
                        &mut activated,
                        &mut output,
                    )?;
                }
                for (actual, reference) in activated.to_host_vec(&stream)?.iter().zip(&expected_act)
                {
                    assert!(
                        (actual - reference).abs() < 1e-4 * reference.abs().max(1.0),
                        "activation {actual} vs {reference}; hidden={hidden}, intermediate={intermediate}"
                    );
                }
                for (actual, reference) in output.to_host_vec(&stream)?.iter().zip(&expected) {
                    assert!(
                        (actual - reference).abs() < 1e-4 * reference.abs().max(1.0),
                        "output {actual} vs {reference}; hidden={hidden}, intermediate={intermediate}"
                    );
                }
                let bits = (
                    activated
                        .to_host_vec(&stream)?
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>(),
                    output
                        .to_host_vec(&stream)?
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>(),
                );
                if let Some(reference) = &bit_reference {
                    for (stage, (a, b)) in [(&bits.0, &reference.0), (&bits.1, &reference.1)]
                        .into_iter()
                        .enumerate()
                    {
                        assert_eq!(a.len(), b.len());
                        if let Some((index, (a, b))) =
                            a.iter().zip(b).enumerate().find(|(_, (a, b))| a != b)
                        {
                            panic!(
                                "staged stage={stage}, index={index}, actual={a:08x}, expected={b:08x}, hidden={hidden}, intermediate={intermediate}, batch={batch}"
                            );
                        }
                    }
                } else {
                    bit_reference = Some(bits);
                }
                if batch <= 8 {
                    let invalid = DeviceBuffer::from_host(&stream, &vec![256u32; batch * 8])?;
                    experts.forward_register(
                        &module,
                        &stream,
                        &work,
                        &x,
                        &invalid,
                        &probs,
                        &mut activated,
                        &mut output,
                    )?;
                    assert!(output.to_host_vec(&stream)?.iter().all(|x| x.is_nan()));
                }
            }
        }
        Ok(())
    }
}
