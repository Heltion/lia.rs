//! Interleaved temporal/row/column RoPE for Qwen 3.5 language attention.
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[kernel]
    pub unsafe fn table(
        batch: usize,
        cache_position: usize,
        prompt_length: usize,
        next_text: u32,
        base: f32,
        positions: &[u32],
        output: *mut f32,
    ) {
        let i = thread::index_1d().get();
        let token = i / 32;
        let pair = i % 32;
        if token < batch {
            // Sections [11,11,10,0] are interleaved, not consecutive bands.
            let axis = pair % 3;
            let index = cache_position + token;
            let position = if index < prompt_length {
                positions[index * 4 + axis]
            } else {
                next_text + (index - prompt_length) as u32
            };
            let angle = position as f32 * base.powf(-(pair as f32) / 32.0);
            unsafe {
                *output.add(token * 64 + pair) = angle.cos();
                *output.add(token * 64 + 32 + pair) = angle.sin();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::vision::{ImageGrid, PositionPlan};
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};

    #[test]
    fn multimodal_rotations_span_images_text_and_draft_positions() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = super::super::load(&context)?;
        let mut plan = PositionPlan::new();
        plan.append_text(3)?;
        plan.append_image(ImageGrid::new(96, 64)?)?;
        plan.append_text(2)?;
        plan.append_image(ImageGrid::new(32, 96)?)?;
        let positions = DeviceBuffer::from_host(&stream, &plan.flattened())?;
        for (start, batch) in [(0, 20), (4, 7), (plan.len(), 8)] {
            let output = DeviceBuffer::<f32>::zeroed(&stream, batch * 64)?;
            // SAFETY: complete four-axis prompt positions, bounded cache indices,
            // and one guarded sine/cosine pair per frequency and token.
            unsafe {
                module.position_table.table(
                    &stream,
                    LaunchConfig::for_num_elems((batch * 32) as u32),
                    batch,
                    start,
                    plan.len(),
                    plan.next_text_position(),
                    10000000.0,
                    &positions,
                    output.cu_deviceptr() as *mut f32,
                )?;
            }
            let actual = output.to_host_vec(&stream)?;
            for t in 0..batch {
                let xyz = plan.coordinates(start + t)?;
                for pair in 0..32 {
                    let axis = [0, 1, 2][pair % 3];
                    let angle = xyz[axis] as f64 * 10000000.0f64.powf(-(pair as f64) / 32.0);
                    assert!((actual[t * 64 + pair] as f64 - angle.cos()).abs() < 3e-6);
                    assert!((actual[t * 64 + 32 + pair] as f64 - angle.sin()).abs() < 3e-6);
                }
            }
        }
        Ok(())
    }
}
