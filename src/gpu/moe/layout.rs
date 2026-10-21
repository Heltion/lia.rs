//! Transactional expert layout changes and format-equivalence validation.
use super::*;

impl Experts {
    pub fn prepare_register(
        &mut self,
        module: &kernels::LoadedModule,
        stream: &std::sync::Arc<CudaStream>,
    ) -> Result<bool> {
        if !(self.gate.can_register() && self.up.can_register() && self.down.can_register()) {
            return Ok(false);
        }
        // Retain all original buffers until all three replacements are ready.
        let gate = self
            .gate
            .repacked(module, stream, 256 * self.intermediate, self.hidden)?;
        let up = self
            .up
            .repacked(module, stream, 256 * self.intermediate, self.hidden)?;
        let down = self
            .down
            .repacked(module, stream, 256 * self.hidden, self.intermediate)?;
        stream.synchronize()?;
        self.gate = gate;
        self.up = up;
        self.down = down;
        Ok(true)
    }

    pub fn is_register(&self) -> bool {
        self.gate.is_register()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: usize, columns: usize, salt: usize) -> Matrix {
        Matrix {
            rows: rows * 256,
            columns,
            bits: 2,
            group_size: 128,
            packed: (0..rows * 256 * columns / 4)
                .map(|i| ((i * 73 + i / 97 + salt) ^ (i >> 5)) as u8)
                .collect(),
            scales: (0..rows * 256 * columns / 128)
                .map(|i| {
                    let exponent = if i % 97 == 0 {
                        3
                    } else {
                        103 + (i * 3 + salt) % 23
                    };
                    f32::from_bits(
                        (((exponent << 7) | ((i * 17) % 128) | ((i + salt) % 2 << 15)) as u32)
                            << 16,
                    )
                })
                .collect(),
        }
    }

    #[test]
    fn grouped_register_layout_is_bitwise_equivalent_at_model_dimensions() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        for (hidden, intermediate, batch) in [(128, 128, 33), (2048, 512, 9), (2048, 512, 33)] {
            let (gate, up, down) = (
                matrix(intermediate, hidden, 17),
                matrix(intermediate, hidden, 59),
                matrix(hidden, intermediate, 97),
            );
            let clone_matrix = |m: &Matrix| Matrix {
                rows: m.rows,
                columns: m.columns,
                bits: m.bits,
                group_size: m.group_size,
                packed: m.packed.clone(),
                scales: m.scales.clone(),
            };
            let original = Experts::upload_offloaded(
                &stream,
                hidden,
                intermediate,
                clone_matrix(&gate),
                clone_matrix(&up),
                clone_matrix(&down),
                0,
            )?;
            let mut reordered =
                Experts::upload_offloaded(&stream, hidden, intermediate, gate, up, down, 0)?;
            assert!(reordered.prepare_register(&module, &stream)?);
            let input = (0..batch * hidden)
                .map(|i| ((i * 19 % 127) as i32 - 63) as f32 / 33.0)
                .collect::<Vec<_>>();
            let ids = (0..batch * 8)
                .map(|i| ((i * 73 + i / 8 * 51) % 256) as u32)
                .collect::<Vec<_>>();
            let probs = (0..batch * 8)
                .map(|i| (i % 8 + 1) as f32 / 36.0)
                .collect::<Vec<_>>();
            let x = DeviceBuffer::from_host(&stream, &input)?;
            let selected = DeviceBuffer::from_host(&stream, &ids)?;
            let probabilities = DeviceBuffer::from_host(&stream, &probs)?;
            let mut work = Grouped::new(&stream, batch)?;
            let mut reference: Option<[Vec<u32>; 3]> = None;
            let packed =
                DeviceBuffer::zeroed(&stream, (batch * hidden).max(batch * 8 * intermediate) / 2)?;
            for static_store in [false, true] {
                work.static_store = static_store;
                for (experts, prepacked, half_codes) in [
                    (&original, false, false),
                    (&reordered, false, false),
                    (&original, true, false),
                    (&reordered, true, false),
                    (&original, false, true),
                    (&reordered, false, true),
                    (&original, true, true),
                    (&reordered, true, true),
                ] {
                    let mut activated = DeviceBuffer::zeroed(&stream, batch * 8 * intermediate)?;
                    let mut choices = DeviceBuffer::zeroed(&stream, batch * 8 * hidden)?;
                    let mut output = DeviceBuffer::zeroed(&stream, batch * hidden)?;
                    experts.forward_grouped(
                        &module,
                        &stream,
                        &work,
                        &x,
                        prepacked.then_some(&packed),
                        half_codes,
                        &selected,
                        &probabilities,
                        &mut activated,
                        &mut choices,
                        &mut output,
                    )?;
                    let stages = [
                        activated.to_host_vec(&stream)?,
                        choices.to_host_vec(&stream)?,
                        output.to_host_vec(&stream)?,
                    ]
                    .map(|v| v.into_iter().map(f32::to_bits).collect::<Vec<_>>());
                    assert!(
                        stages
                            .iter()
                            .flatten()
                            .all(|bits| bits & 0x7f80_0000 != 0x7f80_0000),
                        "synthetic layout validation must produce finite operands/results"
                    );
                    if let Some(expected) = &reference {
                        for (stage, (actual, expected)) in stages.iter().zip(expected).enumerate() {
                            assert_eq!(actual.len(), expected.len());
                            if let Some((index, (a, b))) = actual
                                .iter()
                                .zip(expected)
                                .enumerate()
                                .find(|(_, (a, b))| a != b)
                            {
                                panic!(
                                    "stage={stage}, index={index}, actual={a:08x}, expected={b:08x}, hidden={hidden}, intermediate={intermediate}, batch={batch}"
                                );
                            }
                        }
                    } else {
                        reference = Some(stages);
                    }
                }
            }
        }
        Ok(())
    }
}
