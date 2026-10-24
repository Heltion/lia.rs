//! Qwen vision transformer and merger on the language engine's stream and pool.
use super::*;
use crate::vision::{ImageGrid, PreparedImage};
mod inputs;
mod load;
mod minicpm;
mod minicpm_precision;
mod minicpmo;
pub(in crate::inference) mod offload;
mod operations;

impl Engine {
    /// Encode a frame group in one vision residency interval, preserving the
    /// language cache while parked weights are temporarily uploaded.
    pub fn image_embeddings_many(&mut self, images: &[&PreparedImage]) -> Result<Vec<Vec<f32>>> {
        anyhow::ensure!(
            !images.is_empty() && images.len() <= 32,
            "invalid image slice batch"
        );
        self.with_vision_resident(|engine| {
            images
                .iter()
                .map(|image| {
                    engine
                        .encode_image_resident(image)?
                        .to_host_vec(&engine.stream)
                        .map_err(Into::into)
                })
                .collect()
        })
    }
    /// Diagnostic host export; the multimodal forward path can retain the GPU buffer.
    pub fn image_embeddings(&mut self, image: &PreparedImage) -> Result<Vec<f32>> {
        self.encode_image_buffer(image)?
            .to_host_vec(&self.stream)
            .map_err(Into::into)
    }

    fn encode_image_buffer(&mut self, image: &PreparedImage) -> Result<Buffer> {
        if self.host_vision.is_empty() {
            self.encode_image_resident(image)
        } else {
            self.encode_swapped(image)
        }
    }

    fn encode_image_resident(&self, image: &PreparedImage) -> Result<Buffer> {
        if self.vision_encoder == crate::models::VisionEncoder::MiniCpmO45 {
            return self.encode_minicpmo_image(image);
        }
        if self.vision_encoder == crate::models::VisionEncoder::MiniCpmV47 {
            return self.encode_minicpm_image(image);
        }
        let grid = ImageGrid::new(image.width, image.height)?;
        let patches = grid.token_count() * 4;
        anyhow::ensure!(self.vision_eps.is_some(), "no vision projector loaded");
        anyhow::ensure!(
            patches <= crate::gpu::MAX_PREFILL_BATCH
                && (!self.preallocated || patches <= self.prefill_batch * 4),
            "image exceeds reserved vision workspace"
        );
        anyhow::ensure!(
            image.pixels.len() == image.width * image.height * 3
                && image.pixels.iter().all(|x| x.is_finite()),
            "invalid prepared image pixels"
        );
        let pixels = DeviceBuffer::from_host(&self.stream, &image.pixels)?;
        self.stream.synchronize()?;
        let capture = self.begin_graph()?;
        let result = self.encode_vision(image.width / 16, image.height / 16, &pixels);
        self.finish_graph(capture, (6, patches, false, false), result)
    }

    fn encode_vision(
        &self,
        width: usize,
        height: usize,
        pixels: &DeviceBuffer<f32>,
    ) -> Result<Buffer> {
        let patches = width * height;
        let mut input = self.zeros(patches * 768)?;
        // SAFETY: caller validated complete RGB data and 2x2 patch groups.
        unsafe {
            self.module.vision.patches(
                &self.stream,
                LaunchConfig::for_num_elems(input.len() as u32),
                width * 16,
                height * 16,
                pixels,
                &mut input,
            )?;
        }
        let first = self.vision_mm("v.patch_embd.weight", &input, patches)?;
        let second = self.vision_mm("v.patch_embd.weight.1", &input, patches)?;
        drop(input);
        let mut hidden = self.zeros(patches * 1152)?;
        // SAFETY: two projections contain one 1152-element row per patch;
        // the validated learned position table is 48x48 with matching width.
        unsafe {
            self.module.vision_position.patch_position(
                &self.stream,
                LaunchConfig::for_num_elems(hidden.len() as u32),
                width,
                height,
                &first,
                &second,
                self.floats("vision.v.patch_embd.bias")?,
                self.floats("vision.v.position_embd.weight")?,
                &mut hidden,
            )?;
        }
        drop(first);
        drop(second);
        for layer in 0..27 {
            let p = format!("v.blk.{layer}");
            let normalized = self.vision_norm(&format!("{p}.ln1"), &hidden)?;
            let qkv = self.vision_mm(&format!("{p}.attn_qkv.weight"), &normalized, patches)?;
            drop(normalized);
            let mut rotated = self.zeros(patches * 3456)?;
            // SAFETY: both QKV arrays and bias have the validated head layout.
            unsafe {
                self.module.vision_math.qkv_rope(
                    &self.stream,
                    LaunchConfig::for_num_elems(rotated.len() as u32),
                    width,
                    &qkv,
                    self.floats(&format!("vision.{p}.attn_qkv.bias"))?,
                    &mut rotated,
                )?;
            }
            drop(qkv);
            let attention = self.vision_attention(&rotated, patches, false)?;
            drop(rotated);
            let projected = self.vision_mm(&format!("{p}.attn_out.weight"), &attention, patches)?;
            drop(attention);
            let projected = self.vision_affine(projected, &format!("{p}.attn_out.bias"), false)?;
            let residual = self.add(&hidden, &projected)?;
            drop(hidden);
            drop(projected);
            let normalized = self.vision_norm(&format!("{p}.ln2"), &residual)?;
            let up = self.vision_mm(&format!("{p}.ffn_up.weight"), &normalized, patches)?;
            drop(normalized);
            let up = self.vision_affine(up, &format!("{p}.ffn_up.bias"), true)?;
            let down = self.vision_mm(&format!("{p}.ffn_down.weight"), &up, patches)?;
            drop(up);
            let down = self.vision_affine(down, &format!("{p}.ffn_down.bias"), false)?;
            hidden = self.add(&residual, &down)?;
        }
        let normalized = self.vision_norm("v.post_ln", &hidden)?;
        drop(hidden);
        // Four consecutive encoder patches already form each merger row.
        let merged = self.vision_mm("mm.0.weight", &normalized, patches / 4)?;
        drop(normalized);
        let merged = self.vision_affine(merged, "mm.0.bias", true)?;
        let projected = self.vision_mm("mm.2.weight", &merged, patches / 4)?;
        self.vision_affine(projected, "mm.2.bias", false)
    }
}
