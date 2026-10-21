//! Preserve the checkpoint's BF16 vision operation boundaries when requested.
use super::*;

impl Engine {
    pub(super) fn mini_round(&self, values: Buffer, bf16: bool) -> Result<Buffer> {
        if bf16 {
            // SAFETY: this consumed lease has no concurrent readers or writers.
            unsafe {
                self.module.minicpm_bf16.round(
                    &self.stream,
                    LaunchConfig::for_num_elems(values.len().div_ceil(2) as u32),
                    values.len(),
                    values.cu_deviceptr() as *mut f32,
                )?;
            }
        }
        Ok(values)
    }

    pub(super) fn mini_norm(
        &self,
        prefix: &str,
        values: &DeviceBuffer<f32>,
        bf16: bool,
    ) -> Result<Buffer> {
        self.mini_round(self.vision_norm(prefix, values)?, bf16)
    }

    pub(super) fn mini_add(
        &self,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        bf16: bool,
    ) -> Result<Buffer> {
        self.mini_round(self.add(a, b)?, bf16)
    }

    pub(super) fn mini_affine(
        &self,
        values: Buffer,
        name: &str,
        gelu: bool,
        bf16: bool,
    ) -> Result<Buffer> {
        if !bf16 {
            return self.vision_affine(values, name, gelu);
        }
        let bias = self.floats(&format!("vision.{name}"))?;
        anyhow::ensure!(
            !bias.is_empty() && values.len().is_multiple_of(bias.len()),
            "invalid BF16 vision bias shape"
        );
        // SAFETY: unique output lease and complete rows matching the validated bias.
        unsafe {
            let launch = LaunchConfig::for_num_elems(values.len() as u32);
            if gelu {
                self.module.minicpm_bf16.affine::<true>(
                    &self.stream,
                    launch,
                    values.len(),
                    bias.len(),
                    bias,
                    values.cu_deviceptr() as *mut f32,
                )?;
            } else {
                self.module.minicpm_bf16.affine::<false>(
                    &self.stream,
                    launch,
                    values.len(),
                    bias.len(),
                    bias,
                    values.cu_deviceptr() as *mut f32,
                )?;
            }
        }
        Ok(values)
    }

    pub(super) fn mini_attention(
        &self,
        qkv: &DeviceBuffer<f32>,
        patches: usize,
        window: bool,
        bf16: bool,
    ) -> Result<Buffer> {
        if !bf16 {
            return self.vision_attention(qkv, patches, window);
        }
        anyhow::ensure!(
            patches > 0 && qkv.len() == patches * 3456 && (!window || patches.is_multiple_of(4)),
            "invalid BF16 vision attention extent"
        );
        let attention = self.zeros(patches * 1152)?;
        // SAFETY: width72 heads, bounded windows, and complete guarded query CTAs.
        unsafe {
            let launch = LaunchConfig::for_num_elems((patches * 512) as u32);
            if window {
                self.module.minicpm_bf16.attention::<4>(
                    &self.stream,
                    launch,
                    patches,
                    qkv,
                    attention.cu_deviceptr() as *mut f32,
                )?;
            } else if patches <= 16 {
                self.module.minicpm_bf16.attention::<0>(
                    &self.stream,
                    launch,
                    patches,
                    qkv,
                    attention.cu_deviceptr() as *mut f32,
                )?;
            } else {
                self.module.vision_flash.tiled_attention::<32, true>(
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
        self.mini_round(attention, true)
    }
}
