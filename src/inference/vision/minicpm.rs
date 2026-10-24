//! MiniCPM SigLIP execution with an early2x2 window merger and final projection.
use super::*;

impl Engine {
    pub(super) fn encode_minicpm_image(&self, image: &PreparedImage) -> Result<Buffer> {
        anyhow::ensure!(self.vision_eps.is_some(), "no vision encoder loaded");
        anyhow::ensure!(
            image.width > 0
                && image.height > 0
                && image.width.is_multiple_of(56)
                && image.height.is_multiple_of(56),
            "invalid MiniCPM patch groups"
        );
        let patches = image.width / 14 * (image.height / 14);
        anyhow::ensure!(
            patches <= crate::gpu::MAX_PREFILL_BATCH
                && (!self.preallocated || patches <= self.prefill_batch * 4),
            "image exceeds reserved vision workspace"
        );
        anyhow::ensure!(
            image.pixels.len() == image.width * image.height * 3
                && image.pixels.iter().all(|x| x.is_finite()),
            "invalid MiniCPM RGB pixels"
        );
        let bf16 = std::env::var("LIA_MINICPM_VISION_BF16").as_deref() == Ok("1");
        let pixels = DeviceBuffer::from_host(&self.stream, &image.pixels)?;
        if bf16 {
            // SAFETY: the fresh private pixel allocation has no readers yet.
            unsafe {
                self.module.minicpm_bf16.round(
                    &self.stream,
                    LaunchConfig::for_num_elems(pixels.len().div_ceil(2) as u32),
                    pixels.len(),
                    pixels.cu_deviceptr() as *mut f32,
                )?;
            }
        }
        self.stream.synchronize()?;
        let capture = self.begin_graph()?;
        let result = self.minicpm_inner(image.width / 14, image.height / 14, &pixels, bf16);
        self.finish_graph(capture, (6, patches, false, false), result)
    }

    fn mini_order(&self, width: usize, input: &DeviceBuffer<f32>) -> Result<Buffer> {
        let mut output = self.zeros(input.len())?;
        // SAFETY: callers provide complete1152-channel raster rows and even grids.
        unsafe {
            self.module.minicpm_vision.minicpm_merge_order(
                &self.stream,
                LaunchConfig::for_num_elems(output.len() as u32),
                width,
                input,
                &mut output,
            )?;
        }
        Ok(output)
    }

    fn mini_attention_projection(
        &self,
        prefix: &str,
        input: &DeviceBuffer<f32>,
        patches: usize,
        window: bool,
        bf16: bool,
    ) -> Result<Buffer> {
        let qkv = self.vision_mm(&format!("{prefix}.attn_qkv.weight"), input, patches)?;
        let qkv = self.mini_affine(qkv, &format!("{prefix}.attn_qkv.bias"), false, bf16)?;
        let attention = self.mini_attention(&qkv, patches, window, bf16)?;
        drop(qkv);
        let projected =
            self.vision_mm(&format!("{prefix}.attn_out.weight"), &attention, patches)?;
        self.mini_affine(projected, &format!("{prefix}.attn_out.bias"), false, bf16)
    }

    pub(super) fn mini_block(
        &self,
        prefix: &str,
        hidden: &DeviceBuffer<f32>,
        patches: usize,
        bf16: bool,
    ) -> Result<Buffer> {
        let normalized = self.mini_norm(&format!("{prefix}.ln1"), hidden, bf16)?;
        let attention =
            self.mini_attention_projection(prefix, &normalized, patches, false, bf16)?;
        drop(normalized);
        let residual = self.mini_add(hidden, &attention, bf16)?;
        drop(attention);
        let normalized = self.mini_norm(&format!("{prefix}.ln2"), &residual, bf16)?;
        let up = self.vision_mm(&format!("{prefix}.ffn_up.weight"), &normalized, patches)?;
        drop(normalized);
        let up = self.mini_affine(up, &format!("{prefix}.ffn_up.bias"), true, bf16)?;
        let down = self.vision_mm(&format!("{prefix}.ffn_down.weight"), &up, patches)?;
        drop(up);
        let down = self.mini_affine(down, &format!("{prefix}.ffn_down.bias"), false, bf16)?;
        self.mini_add(&residual, &down, bf16)
    }

    fn mini_insert(
        &self,
        width: usize,
        height: usize,
        hidden: &DeviceBuffer<f32>,
        bf16: bool,
    ) -> Result<Buffer> {
        let patches = width * height;
        let normalized = self.mini_norm("insert.ln1", hidden, bf16)?;
        let ordered = self.mini_order(width, &normalized)?;
        drop(normalized);
        let attention = self.mini_attention_projection("insert", &ordered, patches, true, bf16)?;
        drop(ordered);
        let residual = self.mini_order(width, hidden)?;
        let merged = self.mini_add(&residual, &attention, bf16)?;
        drop(residual);
        drop(attention);
        let mut mean = self.zeros(patches / 4 * 1152)?;
        // SAFETY: four complete adjacent rows per independently merged window.
        unsafe {
            self.module.minicpm_vision.minicpm_window_mean(
                &self.stream,
                LaunchConfig::for_num_elems(mean.len() as u32),
                &merged,
                &mut mean,
            )?;
        }
        let mean = self.mini_round(mean, bf16)?;
        let normalized = self.mini_norm("insert.pre_ln", &merged, bf16)?;
        drop(merged);
        let up = self.vision_mm("insert.up.weight", &normalized, patches / 4)?;
        drop(normalized);
        let up = self.mini_affine(up, "insert.up.bias", true, bf16)?;
        let down = self.vision_mm("insert.down.weight", &up, patches / 4)?;
        drop(up);
        let down = self.mini_affine(down, "insert.down.bias", false, bf16)?;
        self.mini_add(&mean, &down, bf16)
    }

    fn minicpm_inner(
        &self,
        mut width: usize,
        mut height: usize,
        pixels: &DeviceBuffer<f32>,
        bf16: bool,
    ) -> Result<Buffer> {
        let mut patches = width * height;
        let mut input = self.zeros(patches * 588)?;
        // SAFETY: validated RGB planes form complete14x14 patches.
        unsafe {
            self.module.minicpm_vision.minicpm_patches(
                &self.stream,
                LaunchConfig::for_num_elems(input.len() as u32),
                width * 14,
                height * 14,
                pixels,
                &mut input,
            )?;
        }
        let embedded = self.vision_mm("v.patch_embd.weight", &input, patches)?;
        drop(input);
        let mut hidden = self.zeros(patches * 1152)?;
        // SAFETY: model loading validates the70x70 learned table and1152-channel bias.
        unsafe {
            macro_rules! position {
                ($bf16:literal) => {
                    self.module.minicpm_vision.minicpm_position::<$bf16>(
                        &self.stream,
                        LaunchConfig::for_num_elems(hidden.len() as u32),
                        width,
                        height,
                        &embedded,
                        self.floats("vision.v.patch_embd.bias")?,
                        self.floats("vision.v.position_embd.weight")?,
                        &mut hidden,
                    )?;
                };
            }
            if bf16 {
                position!(true);
            } else {
                position!(false);
            }
        }
        drop(embedded);
        for layer in 0..27 {
            hidden = self.mini_block(&format!("v.blk.{layer}"), &hidden, patches, bf16)?;
            if layer == 6 {
                hidden = self.mini_insert(width, height, &hidden, bf16)?;
                width /= 2;
                height /= 2;
                patches /= 4;
            }
        }
        let normalized = self.mini_norm("v.post_ln", &hidden, bf16)?;
        drop(hidden);
        let ordered = self.mini_order(width, &normalized)?;
        drop(normalized);
        let normalized = self.mini_norm("mm.pre_ln", &ordered, bf16)?;
        drop(ordered);
        let up = self.vision_mm("mm.0.weight", &normalized, patches / 4)?;
        drop(normalized);
        // The final projector uses erf-GELU, whereas encoder/insert MLPs use tanh-GELU.
        let bias = self.floats("vision.mm.0.bias")?;
        anyhow::ensure!(
            up.len().is_multiple_of(bias.len()),
            "invalid projector GELU extent"
        );
        // SAFETY: the unique lease covers complete bias rows; no concurrent readers.
        unsafe {
            macro_rules! gelu {
                ($bf16:literal) => {
                    self.module.minicpm_vision.minicpm_gelu_erf::<$bf16>(
                        &self.stream,
                        LaunchConfig::for_num_elems(up.len() as u32),
                        up.len(),
                        bias.len(),
                        bias,
                        up.cu_deviceptr() as *mut f32,
                    )?;
                };
            }
            if bf16 {
                gelu!(true);
            } else {
                gelu!(false);
            }
        }
        let projected = self.vision_mm("mm.2.weight", &up, patches / 4)?;
        self.mini_affine(projected, "mm.2.bias", false, bf16)
    }
}
