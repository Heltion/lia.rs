use super::super::*;

impl Engine {
    pub(super) fn vision_mm(
        &self,
        name: &str,
        input: &DeviceBuffer<f32>,
        batch: usize,
    ) -> Result<Buffer> {
        let compensate = self.vision_encoder == crate::models::VisionEncoder::MiniCpmO45
            && std::env::var("LIA_MINICPM_VISION_BF16").as_deref() != Ok("1");
        self.component_mm(&format!("vision.{name}"), input, batch, compensate)
    }

    pub(super) fn vision_norm(&self, prefix: &str, input: &DeviceBuffer<f32>) -> Result<Buffer> {
        let dimension = self.floats(&format!("vision.{prefix}.weight"))?.len();
        anyhow::ensure!(
            dimension > 0 && input.len().is_multiple_of(dimension),
            "invalid vision normalization input"
        );
        let eps = self
            .vision_eps
            .ok_or_else(|| anyhow!("no vision projector loaded"))?;
        let output = self.zeros(input.len())?;
        // SAFETY: validated row size, complete warps, and separately owned output.
        unsafe {
            self.module.vision_math.layer_norm(
                &self.stream,
                LaunchConfig::for_num_elems((input.len() / dimension * 32) as u32),
                dimension,
                input.len() / dimension,
                eps,
                input,
                self.floats(&format!("vision.{prefix}.weight"))?,
                self.floats(&format!("vision.{prefix}.bias"))?,
                output.cu_deviceptr() as *mut f32,
            )?;
        }
        Ok(output)
    }

    pub(super) fn vision_attention(
        &self,
        qkv: &DeviceBuffer<f32>,
        patches: usize,
        window: bool,
    ) -> Result<Buffer> {
        anyhow::ensure!(
            patches > 0 && qkv.len() == patches * 3456 && (!window || patches.is_multiple_of(4)),
            "invalid vision QKV or window extent"
        );
        let attention = self.zeros(patches * 1152)?;
        // SAFETY: complete16-head width72 QKV rows and the checked4-row windows.
        unsafe {
            if window {
                self.module.vision_attention.attention::<4>(
                    &self.stream,
                    LaunchConfig::for_num_elems((patches * 16 * 32) as u32),
                    patches,
                    qkv,
                    attention.cu_deviceptr() as *mut f32,
                )?;
            } else if patches <= 16 {
                self.module.vision_attention.attention::<0>(
                    &self.stream,
                    LaunchConfig::for_num_elems((patches * 16 * 32) as u32),
                    patches,
                    qkv,
                    attention.cu_deviceptr() as *mut f32,
                )?;
            } else {
                self.module.vision_flash.tiled_attention::<32, false>(
                    &self.stream,
                    LaunchConfig {
                        grid_dim: ((patches.div_ceil(32) * 16) as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    patches,
                    qkv,
                    attention.cu_deviceptr() as *mut f32,
                )?;
            }
        }
        Ok(attention)
    }

    pub(super) fn vision_affine(&self, values: Buffer, name: &str, gelu: bool) -> Result<Buffer> {
        let bias = self.floats(&format!("vision.{name}"))?;
        anyhow::ensure!(
            values.len().is_multiple_of(bias.len()),
            "invalid vision bias shape"
        );
        // SAFETY: the consumed lease uniquely owns all values, and each guarded
        // thread modifies one element using a complete matching bias row.
        unsafe {
            let launch = LaunchConfig::for_num_elems(values.len() as u32);
            let pointer = values.cu_deviceptr() as *mut f32;
            if gelu {
                self.module.vision_math.affine_in_place::<true>(
                    &self.stream,
                    launch,
                    values.len(),
                    bias.len(),
                    bias,
                    pointer,
                )?;
            } else {
                self.module.vision_math.affine_in_place::<false>(
                    &self.stream,
                    launch,
                    values.len(),
                    bias.len(),
                    bias,
                    pointer,
                )?;
            }
        }
        Ok(values)
    }
}
