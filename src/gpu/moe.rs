use super::*;
use crate::quant::gsq::Matrix;
use anyhow::{Result, ensure};
mod grouped;
pub(crate) use grouped::Grouped;

mod storage;
use storage::Packed;

pub(crate) struct Experts {
    hidden: usize,
    intermediate: usize,
    gate: Packed,
    up: Packed,
    down: Packed,
}

impl Experts {
    #[allow(clippy::too_many_arguments)]
    pub fn upload_offloaded(
        stream: &std::sync::Arc<CudaStream>,
        hidden: usize,
        intermediate: usize,
        gate: Matrix,
        up: Matrix,
        down: Matrix,
        offload: usize,
    ) -> Result<Self> {
        ensure!(offload <= 256, "offloaded experts exceed router size");
        ensure!(
            hidden > 0
                && intermediate > 0
                && hidden.is_multiple_of(128)
                && intermediate.is_multiple_of(128),
            "invalid MoE expert dimensions"
        );
        Ok(Self {
            hidden,
            intermediate,
            gate: Packed::upload(stream, gate, 256 * intermediate, hidden, offload)?,
            up: Packed::upload(stream, up, 256 * intermediate, hidden, offload)?,
            down: Packed::upload(stream, down, 256 * hidden, intermediate, offload)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        module: &kernels::LoadedModule,
        stream: &std::sync::Arc<CudaStream>,
        input: &DeviceBuffer<f32>,
        selected: &DeviceBuffer<u32>,
        probabilities: &DeviceBuffer<f32>,
        activated: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if self.gate.is_offloaded() {
            self.forward_impl::<true>(
                module,
                stream,
                input,
                selected,
                probabilities,
                activated,
                output,
            )
        } else {
            self.forward_impl::<false>(
                module,
                stream,
                input,
                selected,
                probabilities,
                activated,
                output,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_impl<const HOST: bool>(
        &self,
        module: &kernels::LoadedModule,
        stream: &std::sync::Arc<CudaStream>,
        input: &DeviceBuffer<f32>,
        selected: &DeviceBuffer<u32>,
        probabilities: &DeviceBuffer<f32>,
        activated: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        ensure!(
            !input.is_empty() && input.len().is_multiple_of(self.hidden),
            "invalid MoE input"
        );
        let batch = input.len() / self.hidden;
        ensure!(
            selected.len() >= batch * 8
                && probabilities.len() >= batch * 8
                && activated.len() == batch * 8 * self.intermediate
                && output.len() == batch * self.hidden,
            "invalid MoE routing or output extent"
        );
        // SAFETY: upload validated packed matrix extents; the live buffers
        // cover all input, routing and output rows on one ordered stream.
        unsafe {
            module.moe.gate_up::<HOST>(
                stream,
                LaunchConfig::for_num_elems((activated.len() * 32) as u32),
                self.hidden,
                self.intermediate,
                selected,
                input,
                &self.gate.data,
                &self.gate.scales,
                &self.up.data,
                &self.up.scales,
                self.gate.host_data(),
                self.gate.host_scales(),
                self.up.host_data(),
                self.up.host_scales(),
                activated.cu_deviceptr() as *mut f32,
            )?;
            module.moe.down::<HOST>(
                stream,
                LaunchConfig::for_num_elems((output.len() * 256) as u32),
                self.hidden,
                self.intermediate,
                selected,
                probabilities,
                activated,
                &self.down.data,
                &self.down.scales,
                self.down.host_data(),
                self.down.host_scales(),
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
    fn routing_and_selected_experts_match_full_scalar_reference() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        let (hidden, intermediate, batch) = (128, 128, 2);
        let matrix = |offset: usize| Matrix {
            rows: 256 * 128,
            columns: 128,
            bits: 2,
            group_size: 128,
            packed: (0usize..256 * 128 * 128 / 4)
                .map(|i| (i * 73 + offset) as u8)
                .collect(),
            scales: (0..256 * 128)
                .map(|i| ((i + offset) % 7) as f32 / 128.0 - 0.03125)
                .collect(),
        };
        let (gate, up, down) = (matrix(17), matrix(59), matrix(97));
        let input = (0..batch * hidden)
            .map(|i| (i % 13) as f32 / 16.0 - 0.3)
            .collect::<Vec<_>>();
        let logits = (0..batch * 256)
            .map(|i| {
                if i < 256 {
                    ((i * 73) % 257) as f32 / 7.0
                } else {
                    -0.5
                }
            })
            .collect::<Vec<_>>();
        let dx = DeviceBuffer::from_host(&stream, &input)?;
        let dl = DeviceBuffer::from_host(&stream, &logits)?;
        let ids = DeviceBuffer::<u32>::zeroed(&stream, batch * 8)?;
        let probabilities = DeviceBuffer::<f32>::zeroed(&stream, batch * 8)?;
        // SAFETY: routing receives complete 256-expert rows and eight slots per token.
        unsafe {
            module.moe.route(
                &stream,
                LaunchConfig::for_num_elems((batch * 32) as u32),
                &dl,
                ids.cu_deviceptr() as *mut u32,
                probabilities.cu_deviceptr() as *mut f32,
            )?;
        }
        let selected = ids.to_host_vec(&stream)?;
        let probs = probabilities.to_host_vec(&stream)?;
        let mut expected_act = vec![0.0f32; batch * 8 * intermediate];
        let mut expected = vec![0.0f32; batch * hidden];
        for token in 0..batch {
            let mut indices = (0..256).collect::<Vec<_>>();
            indices.sort_by(|a, b| {
                logits[token * 256 + *b]
                    .total_cmp(&logits[token * 256 + *a])
                    .then_with(|| a.cmp(b))
            });
            let maximum = logits[token * 256 + indices[0]];
            let masses = indices[..8]
                .iter()
                .map(|id| (logits[token * 256 + id] - maximum).exp())
                .collect::<Vec<_>>();
            let total = masses.iter().sum::<f32>();
            for slot in 0..8 {
                assert_eq!(selected[token * 8 + slot] as usize, indices[slot]);
                assert!((probs[token * 8 + slot] - masses[slot] / total).abs() < 1e-7);
                for row in 0..intermediate {
                    let expert_row = indices[slot] * intermediate + row;
                    let dot = |m: &Matrix| {
                        (0..hidden)
                            .map(|c| m.value(expert_row, c).unwrap() * input[token * hidden + c])
                            .sum::<f32>()
                    };
                    let g = dot(&gate);
                    expected_act[(token * 8 + slot) * intermediate + row] =
                        g / (1.0 + (-g).exp()) * dot(&up);
                }
            }
            for row in 0..hidden {
                for slot in 0..8 {
                    let expert_row = indices[slot] * hidden + row;
                    let sum = (0..intermediate)
                        .map(|c| {
                            down.value(expert_row, c).unwrap()
                                * expected_act[(token * 8 + slot) * intermediate + c]
                        })
                        .sum::<f32>();
                    expected[token * hidden + row] += sum * masses[slot] / total;
                }
            }
        }
        let experts =
            Experts::upload_offloaded(&stream, hidden, intermediate, gate, up, down, 128)?;
        let mut activated = DeviceBuffer::zeroed(&stream, expected_act.len())?;
        let mut output = DeviceBuffer::zeroed(&stream, expected.len())?;
        experts.forward(
            &module,
            &stream,
            &dx,
            &ids,
            &probabilities,
            &mut activated,
            &mut output,
        )?;
        for (actual, expected) in activated.to_host_vec(&stream)?.iter().zip(expected_act) {
            assert!((actual - expected).abs() < 1e-5 * expected.abs().max(1.0));
        }
        for (actual, expected) in output.to_host_vec(&stream)?.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5 * expected.abs().max(1.0));
        }
        Ok(())
    }
}
