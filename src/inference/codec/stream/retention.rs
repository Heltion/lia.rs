//! Match the pinned reference's packed encoder and current-first CFM retention.
use super::*;
use crate::models::minicpm_o45::stream_layout;
impl Engine {
    pub fn trim_codec_encoder_cache(
        &self,
        cache: &mut CodecEncoderCache,
        prefix_frames: usize,
        suffix_frames: usize,
    ) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&cache.context, &self._context) && cache.valid,
            "invalid encoder cache for retention"
        );
        let low = cache.history_codes();
        let high = cache.history_frames();
        ensure!(high == low * 2, "inconsistent two-rate encoder cache");
        let plan = stream_layout::encoder_retention(low, prefix_frames, suffix_frames)?;
        if plan.high.len() == high {
            return Ok(());
        }
        ensure!(
            cache.attention.iter().enumerate().all(|(i, v)| v
                .as_ref()
                .is_some_and(|v| v.len() == if i < 6 { low * 1024 } else { high * 1024 })),
            "inconsistent encoder layer history"
        );
        let low_indices = self.stream_indices(&plan.low, low)?;
        let high_indices = self.stream_indices(&plan.high, high)?;
        cache.valid = false;
        for (index, layer) in cache.attention.iter_mut().enumerate() {
            let old = layer
                .as_deref()
                .ok_or_else(|| anyhow!("missing encoder history"))?;
            *layer = Some(self.stream_select(
                old,
                1,
                if index < 6 { low } else { high },
                if index < 6 {
                    &low_indices
                } else {
                    &high_indices
                },
            )?);
        }
        cache.valid = true;
        Ok(())
    }
    pub fn trim_codec_flow_cache(
        &self,
        cache: &mut CodecFlowCache,
        prefix_frames: usize,
        suffix_frames: usize,
    ) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&cache.context, &self._context) && cache.valid,
            "invalid flow cache for retention"
        );
        let frames = cache.history_frames();
        let plan = stream_layout::retention_indices(frames, prefix_frames, suffix_frames)?;
        if plan.len() == frames {
            return Ok(());
        }
        ensure!(
            cache.layers.iter().flatten().all(|v| v
                .attention
                .as_ref()
                .is_some_and(|v| v.len() == 2 * frames * 1024)),
            "inconsistent flow layer history"
        );
        let indices = self.stream_indices(&plan, frames)?;
        cache.valid = false;
        for layer in cache.layers.iter_mut().flatten() {
            let old = layer
                .attention
                .as_deref()
                .ok_or_else(|| anyhow!("missing flow history"))?;
            layer.attention = Some(self.stream_select(old, 2, frames, &indices)?);
        }
        cache.valid = true;
        Ok(())
    }
    fn stream_indices(&self, values: &[u32], frames: usize) -> Result<DeviceBuffer<u32>> {
        ensure!(
            !values.is_empty() && values.iter().all(|&i| (i as usize) < frames),
            "invalid acoustic cache indices"
        );
        Ok(DeviceBuffer::from_host(&self.stream, values)?)
    }
    fn stream_select(
        &self,
        input: &DeviceBuffer<f32>,
        samples: usize,
        frames: usize,
        indices: &DeviceBuffer<u32>,
    ) -> Result<Buffer> {
        ensure!(
            samples > 0
                && frames > 0
                && input.len() == samples * frames * 1024
                && !indices.is_empty(),
            "invalid acoustic selection extents"
        );
        let mut output = self.cache_buffer(samples * indices.len() * 1024)?;
        // SAFETY: CPU-validated frame indices gather complete rows within each isolated sample.
        unsafe {
            self.module.codec_stream.stream_select(
                &self.stream,
                LaunchConfig::for_num_elems(output.len() as u32),
                frames,
                indices,
                input,
                &mut output,
            )?;
        }
        Ok(output)
    }
}
