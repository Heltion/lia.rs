use super::*;

pub(crate) struct Grouped {
    pub(super) static_store: bool,
    capacity: usize,
    counts: DeviceBuffer<u32>,
    choices: DeviceBuffer<u32>,
    tiles: DeviceBuffer<u32>,
}

impl Grouped {
    pub fn new(stream: &std::sync::Arc<CudaStream>, capacity: usize) -> Result<Self> {
        ensure!(
            (1..=MAX_PREFILL_BATCH).contains(&capacity),
            "invalid grouped MoE capacity"
        );
        Ok(Self {
            static_store: std::env::var("LIA_MOE_STATIC_STORE").as_deref() == Ok("1"),
            capacity,
            counts: DeviceBuffer::zeroed(stream, 256)?,
            choices: DeviceBuffer::zeroed(stream, 256 * capacity)?,
            tiles: DeviceBuffer::zeroed(stream, 1 + 2 * (capacity.div_ceil(4) + 256))?,
        })
    }
}

impl Experts {
    /// The selected IDs must be eight distinct, bounded router outputs per token.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_grouped(
        &self,
        module: &kernels::LoadedModule,
        stream: &std::sync::Arc<CudaStream>,
        work: &Grouped,
        input: &DeviceBuffer<f32>,
        packed_input: Option<&DeviceBuffer<u32>>,
        half_codes: bool,
        selected: &DeviceBuffer<u32>,
        probabilities: &DeviceBuffer<f32>,
        activated: &mut DeviceBuffer<f32>,
        choices_out: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if self.is_register() {
            return self.forward_grouped_impl::<false, true>(
                module,
                stream,
                work,
                input,
                packed_input,
                half_codes,
                selected,
                probabilities,
                activated,
                choices_out,
                output,
            );
        }
        if self.gate.is_offloaded() {
            self.forward_grouped_impl::<true, false>(
                module,
                stream,
                work,
                input,
                packed_input,
                half_codes,
                selected,
                probabilities,
                activated,
                choices_out,
                output,
            )
        } else {
            self.forward_grouped_impl::<false, false>(
                module,
                stream,
                work,
                input,
                packed_input,
                half_codes,
                selected,
                probabilities,
                activated,
                choices_out,
                output,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_grouped_impl<const HOST: bool, const REGISTER: bool>(
        &self,
        module: &kernels::LoadedModule,
        stream: &std::sync::Arc<CudaStream>,
        work: &Grouped,
        input: &DeviceBuffer<f32>,
        packed_input: Option<&DeviceBuffer<u32>>,
        half_codes: bool,
        selected: &DeviceBuffer<u32>,
        probabilities: &DeviceBuffer<f32>,
        activated: &mut DeviceBuffer<f32>,
        choices_out: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        ensure!(
            !input.is_empty() && input.len().is_multiple_of(self.hidden),
            "invalid grouped expert input"
        );
        let batch = input.len() / self.hidden;
        ensure!(
            batch <= work.capacity
                && selected.len() >= batch * 8
                && probabilities.len() >= batch * 8
                && activated.len() == batch * 8 * self.intermediate
                && choices_out.len() == batch * 8 * self.hidden
                && output.len() == batch * self.hidden,
            "invalid grouped expert buffer extents"
        );
        let maximum_tiles = batch.div_ceil(4) + 256;
        // SAFETY: the router produces distinct IDs in 0..256. Each expert list
        // has at most batch choices. The sum of ceil(count/32) is bounded by
        // batch*8/32 + 256; descriptor and MMA launches are stream ordered.
        unsafe {
            module.moe_grouping.group(
                stream,
                LaunchConfig {
                    grid_dim: (64, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                },
                batch,
                selected,
                work.counts.cu_deviceptr() as *mut u32,
                work.choices.cu_deviceptr() as *mut u32,
            )?;
            module.moe_grouping.descriptors(
                stream,
                LaunchConfig::for_num_elems(1),
                &work.counts,
                work.tiles.cu_deviceptr() as *mut u32,
            )?;
            let config = |rows: usize| LaunchConfig {
                grid_dim: ((rows.div_ceil(64) * maximum_tiles) as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            if let Some(packed) = packed_input {
                ensure!(
                    packed.len() >= input.len().max(activated.len()) / 2,
                    "packed grouped expert workspace too small"
                );
            }
            macro_rules! project {
                ($gate:literal, $packed:literal, $weights:expr, $up:expr, $x:expr, $out:expr) => {
                    macro_rules! run {
                        ($half:literal, $static:literal) => {
                            module
                                .moe_tensor
                                .matmul::<$gate, HOST, REGISTER, $packed, $half, $static>(
                                    stream,
                                    config(if $gate {
                                        self.intermediate
                                    } else {
                                        self.hidden
                                    }),
                                    if $gate {
                                        self.hidden
                                    } else {
                                        self.intermediate
                                    },
                                    if $gate {
                                        self.intermediate
                                    } else {
                                        self.hidden
                                    },
                                    batch,
                                    &work.counts,
                                    &work.choices,
                                    &work.tiles,
                                    &$weights.data,
                                    &$weights.scales,
                                    &$up.data,
                                    &$up.scales,
                                    $weights.host_data(),
                                    $weights.host_scales(),
                                    $up.host_data(),
                                    $up.host_scales(),
                                    $x,
                                    packed_input.unwrap_or(&work.tiles),
                                    $out.cu_deviceptr() as *mut f32,
                                )?
                        };
                    }
                    match (half_codes, work.static_store) {
                        (false, false) => run!(false, false),
                        (true, false) => run!(true, false),
                        (false, true) => run!(false, true),
                        (true, true) => run!(true, true),
                    }
                };
            }
            if let Some(packed) = packed_input {
                module.tensor.pack_input(
                    stream,
                    LaunchConfig::for_num_elems((input.len() / 2) as u32),
                    input,
                    packed.cu_deviceptr() as *mut u32,
                )?;
                project!(true, true, self.gate, self.up, input, activated);
                module.tensor.pack_input(
                    stream,
                    LaunchConfig::for_num_elems((activated.len() / 2) as u32),
                    activated,
                    packed.cu_deviceptr() as *mut u32,
                )?;
                project!(false, true, self.down, self.down, activated, choices_out);
            } else {
                project!(true, false, self.gate, self.up, input, activated);
                project!(false, false, self.down, self.down, activated, choices_out);
            }
            module.moe_grouping.merge(
                stream,
                LaunchConfig::for_num_elems(output.len() as u32),
                batch,
                self.hidden,
                choices_out,
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
    fn grouped_experts_match_independent_half_operand_reference() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        let matrix = |offset: usize| Matrix {
            rows: 256 * 128,
            columns: 128,
            bits: 2,
            group_size: 128,
            packed: (0..256 * 128 * 128 / 4)
                .map(|i| (i * 73 + offset) as u8)
                .collect(),
            scales: (0..256 * 128)
                .map(|i| ((i + offset) % 7) as f32 / 128.0 - 0.03125)
                .collect(),
        };
        let (gate, up, down) = (matrix(17), matrix(59), matrix(97));
        let clone_matrix = |m: &Matrix| Matrix {
            rows: m.rows,
            columns: m.columns,
            bits: m.bits,
            group_size: m.group_size,
            packed: m.packed.clone(),
            scales: m.scales.clone(),
        };
        for offload in [0, 128] {
            let experts = Experts::upload_offloaded(
                &stream,
                128,
                128,
                clone_matrix(&gate),
                clone_matrix(&up),
                clone_matrix(&down),
                offload,
            )?;
            let work = Grouped::new(&stream, 257)?;
            for batch in [9usize, 33, 129, 257] {
                let selected = (0..batch * 8)
                    .map(|i| {
                        if batch < 100 {
                            (i % 8) as u32
                        } else if batch == 257 {
                            (248 + i % 8) as u32
                        } else {
                            ((i / 8 * 17 + i % 8 * 31) % 256) as u32
                        }
                    })
                    .collect::<Vec<_>>();
                let probabilities = (0..batch * 8)
                    .map(|i| (i % 8 + 1) as f32 / 36.0)
                    .collect::<Vec<_>>();
                let input = (0..batch * 128)
                    .map(|i| (i as i32 % 19 - 9) as f32 / 64.0)
                    .collect::<Vec<_>>();
                let mut expected_act = vec![0.0f32; batch * 8 * 128];
                let mut expected = vec![0.0f32; batch * 128];
                for choice in 0..batch * 8 {
                    let token = choice / 8;
                    for row in 0..128 {
                        let dot = |matrix: &Matrix| {
                            (0..128)
                                .map(|c| {
                                    let w = (matrix
                                        .value(selected[choice] as usize * 128 + row, c)
                                        .unwrap()
                                        as f16) as f32;
                                    let x = (input[token * 128 + c] as f16) as f32;
                                    w * x
                                })
                                .sum::<f32>()
                        };
                        let g = dot(&gate);
                        expected_act[choice * 128 + row] = g / (1.0 + (-g).exp()) * dot(&up);
                    }
                    for row in 0..128 {
                        let sum = (0..128)
                            .map(|c| {
                                let w = (down
                                    .value(selected[choice] as usize * 128 + row, c)
                                    .unwrap() as f16)
                                    as f32;
                                w * ((expected_act[choice * 128 + c] as f16) as f32)
                            })
                            .sum::<f32>();
                        expected[token * 128 + row] += sum * probabilities[choice];
                    }
                }
                let dx = DeviceBuffer::from_host(&stream, &input)?;
                let ids = DeviceBuffer::from_host(&stream, &selected)?;
                let probs = DeviceBuffer::from_host(&stream, &probabilities)?;
                let mut activated = DeviceBuffer::zeroed(&stream, expected_act.len())?;
                let mut choices = DeviceBuffer::zeroed(&stream, batch * 8 * 128)?;
                let mut output = DeviceBuffer::zeroed(&stream, expected.len())?;
                experts.forward_grouped(
                    &module,
                    &stream,
                    &work,
                    &dx,
                    None,
                    false,
                    &ids,
                    &probs,
                    &mut activated,
                    &mut choices,
                    &mut output,
                )?;
                let baseline = [
                    activated.to_host_vec(&stream)?,
                    choices.to_host_vec(&stream)?,
                    output.to_host_vec(&stream)?,
                ];
                let extent = dx.len().max(activated.len()) / 2;
                let sentinel = 0x1234_5678u32;
                let packed = DeviceBuffer::from_host(&stream, &vec![sentinel; extent + 32])?;
                experts.forward_grouped(
                    &module,
                    &stream,
                    &work,
                    &dx,
                    Some(&packed),
                    true,
                    &ids,
                    &probs,
                    &mut activated,
                    &mut choices,
                    &mut output,
                )?;
                for (actual, expected) in [
                    activated.to_host_vec(&stream)?,
                    choices.to_host_vec(&stream)?,
                    output.to_host_vec(&stream)?,
                ]
                .iter()
                .zip(&baseline)
                {
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "packed grouped result: batch={batch}, offload={offload}"
                    );
                }
                assert!(
                    packed.to_host_vec(&stream)?[extent..]
                        .iter()
                        .all(|&x| x == sentinel)
                );
                for (actual, expected) in activated.to_host_vec(&stream)?.iter().zip(expected_act) {
                    assert!(
                        (actual - expected).abs() < 1e-4 * expected.abs().max(1.0),
                        "activation batch={batch}"
                    );
                }
                for (actual, expected) in output.to_host_vec(&stream)?.iter().zip(expected) {
                    assert!(
                        (actual - expected).abs() < 1e-4 * expected.abs().max(1.0),
                        "output batch={batch}: {actual} != {expected}"
                    );
                }
                let counts = work.counts.to_host_vec(&stream)?;
                let mapping = work.choices.to_host_vec(&stream)?;
                let tiles = work.tiles.to_host_vec(&stream)?;
                let mut expected_tiles = 0;
                for expert in 0..256 {
                    let expected = selected
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &id)| (id as usize == expert).then_some(i as u32))
                        .collect::<Vec<_>>();
                    assert_eq!(counts[expert] as usize, expected.len());
                    assert_eq!(
                        &mapping[expert * batch..expert * batch + expected.len()],
                        expected
                    );
                    expected_tiles += expected.len().div_ceil(32);
                }
                assert_eq!(tiles[0] as usize, expected_tiles);
            }
        }
        Ok(())
    }
}
