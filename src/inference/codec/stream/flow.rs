//! Chunked Euler integration retains the distinct DiT state of each timestep.
use super::*;
impl Engine {
    pub fn new_codec_flow_cache(&self, steps: usize) -> Result<CodecFlowCache> {
        ensure!(
            (1..=64).contains(&steps)
                && self
                    .weights
                    .contains_key("codec.decoder.estimator.final_layer.linear.weight"),
            "invalid flow step count or unloaded weights"
        );
        let mut layers = Vec::with_capacity(steps);
        for _ in 0..steps {
            let mut step = Vec::with_capacity(16);
            for _ in 0..16 {
                step.push(FlowLayer {
                    attention: None,
                    cnn1: self.stream_zeroed(2 * 2 * 512)?,
                    cnn2: self.stream_zeroed(2 * 2 * 512)?,
                });
            }
            layers.push(step);
        }
        Ok(CodecFlowCache {
            context: self._context.clone(),
            position: 0,
            valid: true,
            layers,
        })
    }
    pub fn codec_flow_chunk(
        &self,
        cache: &mut CodecFlowCache,
        noise: &[f32],
        mu: &[f32],
        speaker: &[f32],
        condition: &[f32],
    ) -> Result<Vec<f32>> {
        self.cfm_validate(noise, mu, speaker, condition)?;
        let noise = DeviceBuffer::from_host(&self.stream, noise)?;
        let mu = DeviceBuffer::from_host(&self.stream, mu)?;
        let speaker = DeviceBuffer::from_host(&self.stream, speaker)?;
        let condition = DeviceBuffer::from_host(&self.stream, condition)?;
        self.codec_flow_chunk_device(cache, &noise, &mu, &speaker, &condition)?
            .to_host_vec(&self.stream)
            .map_err(Into::into)
    }
    pub(in crate::inference::codec) fn codec_flow_chunk_device(
        &self,
        cache: &mut CodecFlowCache,
        noise: &DeviceBuffer<f32>,
        mu: &DeviceBuffer<f32>,
        speaker: &DeviceBuffer<f32>,
        condition: &DeviceBuffer<f32>,
    ) -> Result<Buffer> {
        ensure!(
            Arc::ptr_eq(&cache.context, &self._context) && cache.valid,
            "invalid or poisoned flow cache"
        );
        ensure!(
            !noise.is_empty()
                && noise.len().is_multiple_of(80)
                && mu.len() == noise.len()
                && condition.len() == noise.len()
                && speaker.len() == 80,
            "invalid chunked flow extents"
        );
        let frames = noise.len() / 80;
        ensure!(
            frames <= crate::models::minicpm_o45::MAX_MEL_FRAMES.saturating_sub(cache.position),
            "flow stream exceeds frame capacity"
        );
        cache.valid = false;
        let mut x = self.zeros(noise.len())?;
        x.copy_from_device_async(noise, &self.stream)?;
        let steps = cache.layers.len();
        let mut time = 0.;
        for step in 0..steps {
            let end =
                1. - (((step + 1) as f32 / steps as f32) * core::f32::consts::FRAC_PI_2).cos();
            let velocity = self.cfm_stream_velocity(
                &x,
                mu,
                speaker,
                condition,
                frames,
                time,
                &mut cache.layers[step],
            )?;
            // SAFETY: two isolated CFG velocity rows update one complete uniquely owned state.
            unsafe {
                self.module.cfm_math.cfm_euler(
                    &self.stream,
                    LaunchConfig::for_num_elems(x.len() as u32),
                    end - time,
                    &velocity,
                    x.len(),
                    x.cu_deviceptr() as *mut f32,
                )?;
            }
            time = end;
        }
        cache.position += frames;
        cache.valid = true;
        Ok(x)
    }
    fn cfm_stream_velocity(
        &self,
        x: &DeviceBuffer<f32>,
        mu: &DeviceBuffer<f32>,
        speaker: &DeviceBuffer<f32>,
        condition: &DeviceBuffer<f32>,
        frames: usize,
        time: f32,
        layers: &mut [FlowLayer],
    ) -> Result<Buffer> {
        ensure!(layers.len() == 16, "invalid flow layer cache count");
        let (mut hidden, time) = self.cfm_prepare(x, mu, speaker, condition, frames, time)?;
        for (index, layer) in layers.iter_mut().enumerate() {
            hidden = self.cfm_chunk_block(
                &format!("decoder.estimator.blocks.{index}"),
                &hidden,
                &time,
                frames,
                layer,
            )?;
        }
        self.cfm_finalize(&hidden, &time, frames)
    }
}
