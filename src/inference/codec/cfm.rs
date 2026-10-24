//! Conditional DiT flow matching, with independent CFG samples on the GPU.
use super::*;
mod input;

impl Engine {
    /// Evaluate conditional and unconditional velocities, each in time-major order.
    pub fn codec_dit_velocity(
        &self,
        x: &[f32],
        mu: &[f32],
        speaker: &[f32],
        condition: &[f32],
        time: f32,
    ) -> Result<Vec<f32>> {
        let frames = self.cfm_validate(x, mu, speaker, condition)?;
        ensure!(
            time.is_finite() && (0.0..=1.0).contains(&time),
            "invalid flow timestep"
        );
        let x = DeviceBuffer::from_host(&self.stream, x)?;
        let mu = DeviceBuffer::from_host(&self.stream, mu)?;
        let speaker = DeviceBuffer::from_host(&self.stream, speaker)?;
        let condition = DeviceBuffer::from_host(&self.stream, condition)?;
        self.cfm_velocity(&x, &mu, &speaker, &condition, frames, time)?
            .to_host_vec(&self.stream)
            .map_err(Into::into)
    }
    /// Integrate the cosine-scheduled Euler flow from explicit initial noise.
    pub fn codec_cfm_mel(
        &self,
        noise: &[f32],
        mu: &[f32],
        speaker: &[f32],
        condition: &[f32],
        steps: usize,
    ) -> Result<Vec<f32>> {
        let frames = self.cfm_validate(noise, mu, speaker, condition)?;
        ensure!((1..=64).contains(&steps), "invalid flow step count");
        let mut x = self.zeros(noise.len())?;
        x.copy_from_host(&self.stream, noise)?;
        let mu = DeviceBuffer::from_host(&self.stream, mu)?;
        let speaker = DeviceBuffer::from_host(&self.stream, speaker)?;
        let condition = DeviceBuffer::from_host(&self.stream, condition)?;
        self.cfm_integrate(x, &mu, &speaker, &condition, frames, steps)?
            .to_host_vec(&self.stream)
            .map_err(Into::into)
    }
    pub(super) fn cfm_integrate(
        &self,
        x: Buffer,
        mu: &DeviceBuffer<f32>,
        speaker: &DeviceBuffer<f32>,
        condition: &DeviceBuffer<f32>,
        frames: usize,
        steps: usize,
    ) -> Result<Buffer> {
        ensure!(
            (1..=64).contains(&steps)
                && frames > 0
                && x.len() == frames * 80
                && mu.len() == x.len()
                && condition.len() == x.len()
                && speaker.len() == 80,
            "invalid device flow extents"
        );
        let mut time = 0.0;
        for step in 0..steps {
            let end =
                1.0 - (((step + 1) as f32 / steps as f32) * core::f32::consts::FRAC_PI_2).cos();
            let dt = end - time;
            let velocity = self.cfm_velocity(&x, &mu, &speaker, &condition, frames, time)?;
            // SAFETY: the uniquely owned state has exactly one sample; velocity contains two.
            unsafe {
                self.module.cfm_math.cfm_euler(
                    &self.stream,
                    LaunchConfig::for_num_elems(x.len() as u32),
                    dt,
                    &velocity,
                    x.len(),
                    x.cu_deviceptr() as *mut f32,
                )?;
            }
            time += dt;
        }
        Ok(x)
    }
    pub(super) fn cfm_validate(
        &self,
        x: &[f32],
        mu: &[f32],
        speaker: &[f32],
        condition: &[f32],
    ) -> Result<usize> {
        ensure!(
            self.weights
                .contains_key("codec.decoder.estimator.final_layer.linear.weight"),
            "flow weights are not loaded"
        );
        ensure!(
            !x.is_empty()
                && x.len().is_multiple_of(80)
                && x.len() <= crate::models::minicpm_o45::MAX_MEL_FRAMES * 80
                && mu.len() == x.len()
                && condition.len() == x.len()
                && speaker.len() == 80,
            "invalid flow conditioning dimensions"
        );
        ensure!(
            x.iter()
                .chain(mu)
                .chain(speaker)
                .chain(condition)
                .all(|v| v.is_finite()),
            "nonfinite flow conditioning"
        );
        Ok(x.len() / 80)
    }
    fn cfm_velocity(
        &self,
        x: &DeviceBuffer<f32>,
        mu: &DeviceBuffer<f32>,
        speaker: &DeviceBuffer<f32>,
        condition: &DeviceBuffer<f32>,
        frames: usize,
        time: f32,
    ) -> Result<Buffer> {
        let (mut hidden, time) = self.cfm_prepare(x, mu, speaker, condition, frames, time)?;
        let root = "decoder.estimator";
        for layer in 0..16 {
            hidden = self.cfm_block(&format!("{root}.blocks.{layer}"), &hidden, &time, frames)?;
        }
        self.cfm_finalize(&hidden, &time, frames)
    }
    fn cfm_block(
        &self,
        name: &str,
        input: &DeviceBuffer<f32>,
        time: &DeviceBuffer<f32>,
        frames: usize,
    ) -> Result<Buffer> {
        let rows = frames * 2;
        let modulation = self.codec_linear(
            &format!("{name}.adaLN_modulation.1"),
            time,
            1,
            Activation::Linear,
        )?;
        let normalized = self.cfm_modulate(input, &modulation, 0)?;
        let qkv = self.codec_linear(
            &format!("{name}.qkv"),
            &normalized,
            rows,
            Activation::Linear,
        )?;
        let attention = self.zeros(rows * 512)?;
        // SAFETY: full width-64 heads; both CFG samples are isolated by their frame boundary.
        unsafe {
            self.module.cfm_attention.cfm_qk_norm(
                &self.stream,
                LaunchConfig::for_num_elems((rows * 16 * 32) as u32),
                rows,
                self.floats(&format!("codec.{name}.attn.q_norm.weight"))?,
                self.floats(&format!("codec.{name}.attn.q_norm.bias"))?,
                self.floats(&format!("codec.{name}.attn.k_norm.weight"))?,
                self.floats(&format!("codec.{name}.attn.k_norm.bias"))?,
                qkv.cu_deviceptr() as *mut f32,
            )?;
            self.module.cfm_attention.cfm_attention(
                &self.stream,
                LaunchConfig::for_num_elems((rows * 8 * 32) as u32),
                frames,
                &qkv,
                attention.cu_deviceptr() as *mut f32,
            )?;
        }
        let branch = self.codec_linear(
            &format!("{name}.attn.proj"),
            &attention,
            rows,
            Activation::Linear,
        )?;
        let hidden = self.cfm_add(input, &branch, &modulation, 2)?;
        let normalized = self.cfm_modulate(&hidden, &modulation, 6)?;
        let conv = self.cfm_conv(&format!("{name}.conv.block.1"), &normalized, frames)?;
        let conv = self.codec_norm(&format!("{name}.conv.block.3"), &conv, 1e-5)?;
        self.cfm_activate(&conv, 1)?;
        let conv = self.cfm_conv(&format!("{name}.conv.block.6"), &conv, frames)?;
        let hidden = self.cfm_add(&hidden, &conv, &modulation, 8)?;
        let normalized = self.cfm_modulate(&hidden, &modulation, 3)?;
        let up = self.codec_linear(
            &format!("{name}.mlp.fc1"),
            &normalized,
            rows,
            Activation::Gelu,
        )?;
        let down = self.codec_linear(&format!("{name}.mlp.fc2"), &up, rows, Activation::Linear)?;
        self.cfm_add(&hidden, &down, &modulation, 5)
    }
    pub(super) fn cfm_modulate(
        &self,
        input: &DeviceBuffer<f32>,
        modulation: &DeviceBuffer<f32>,
        component: usize,
    ) -> Result<Buffer> {
        let out = self.zeros(input.len())?;
        // SAFETY: complete 512-channel rows with the matching shift/scale pair.
        unsafe {
            self.module.cfm_math.cfm_modulate(
                &self.stream,
                LaunchConfig::for_num_elems((input.len() / 512 * 32) as u32),
                component,
                input,
                modulation,
                out.cu_deviceptr() as *mut f32,
            )?;
        }
        Ok(out)
    }
    pub(super) fn cfm_add(
        &self,
        input: &DeviceBuffer<f32>,
        branch: &DeviceBuffer<f32>,
        modulation: &DeviceBuffer<f32>,
        component: usize,
    ) -> Result<Buffer> {
        let out = self.zeros(input.len())?;
        // SAFETY: matching residual rows and a complete per-channel gate.
        unsafe {
            self.module.cfm_math.cfm_gated_add(
                &self.stream,
                LaunchConfig::for_num_elems(input.len() as u32),
                component,
                input,
                branch,
                modulation,
                out.cu_deviceptr() as *mut f32,
            )?;
        }
        Ok(out)
    }
    pub(super) fn cfm_activate(&self, input: &DeviceBuffer<f32>, mode: usize) -> Result<()> {
        // SAFETY: callers provide a uniquely owned intermediate, never an engine weight.
        unsafe {
            self.module.cfm_math.cfm_activation(
                &self.stream,
                LaunchConfig::for_num_elems(input.len() as u32),
                mode,
                input.len(),
                input.cu_deviceptr() as *mut f32,
            )?;
        }
        Ok(())
    }
    fn cfm_conv(&self, name: &str, input: &DeviceBuffer<f32>, frames: usize) -> Result<Buffer> {
        ensure!(
            input.len() == frames * 2 * 512,
            "invalid CFG convolution rows"
        );
        let mut columns = self.zeros(frames * 2 * 512 * 3)?;
        // SAFETY: each of the two samples is padded independently by the kernel.
        unsafe {
            self.module.codec.codec_columns(
                &self.stream,
                LaunchConfig::for_num_elems(columns.len() as u32),
                frames,
                3,
                2,
                1,
                input,
                &mut columns,
            )?;
        }
        self.codec_linear(name, &columns, frames * 2, Activation::Linear)
    }
}
