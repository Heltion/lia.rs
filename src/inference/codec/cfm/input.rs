//! Shared timestep/CFG packing and final projection for full and chunked flow.
use super::*;
impl Engine {
    pub(in crate::inference::codec) fn cfm_prepare(
        &self,
        x: &DeviceBuffer<f32>,
        mu: &DeviceBuffer<f32>,
        speaker: &DeviceBuffer<f32>,
        condition: &DeviceBuffer<f32>,
        frames: usize,
        time: f32,
    ) -> Result<(Buffer, Buffer)> {
        let mut packed = self.zeros(frames * 2 * 320)?;
        let mut timestep = self.zeros(256)?;
        // SAFETY: complete validated input rows and disjoint output buffers.
        unsafe {
            self.module.cfm_math.cfm_pack(
                &self.stream,
                LaunchConfig::for_num_elems(packed.len() as u32),
                frames,
                x,
                mu,
                speaker,
                condition,
                &mut packed,
            )?;
            self.module.cfm_math.cfm_time(
                &self.stream,
                LaunchConfig::for_num_elems(256),
                time,
                &mut timestep,
            )?;
        }
        let root = "decoder.estimator";
        let time = self.codec_linear(
            &format!("{root}.t_embedder.mlp.0"),
            &timestep,
            1,
            Activation::Silu,
        )?;
        let time = self.codec_linear(
            &format!("{root}.t_embedder.mlp.2"),
            &time,
            1,
            Activation::Linear,
        )?;
        self.cfm_activate(&time, 0)?;
        let hidden = self.codec_linear(
            &format!("{root}.in_proj"),
            &packed,
            frames * 2,
            Activation::Linear,
        )?;
        Ok((hidden, time))
    }
    pub(in crate::inference::codec) fn cfm_finalize(
        &self,
        hidden: &DeviceBuffer<f32>,
        time: &DeviceBuffer<f32>,
        frames: usize,
    ) -> Result<Buffer> {
        let root = "decoder.estimator";
        let modulation = self.codec_linear(
            &format!("{root}.final_layer.adaLN_modulation.1"),
            time,
            1,
            Activation::Linear,
        )?;
        let normalized = self.cfm_modulate(hidden, &modulation, 0)?;
        self.codec_linear(
            &format!("{root}.final_layer.linear"),
            &normalized,
            frames * 2,
            Activation::Linear,
        )
    }
}
