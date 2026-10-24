//! MiniCPM-o SigLIP without the V4.7 merger, followed by the perceiver resampler.
use super::*;

impl Engine {
    pub(super) fn encode_minicpmo_image(&self, image: &PreparedImage) -> Result<Buffer> {
        let encoded = self.minicpmo_siglip(image)?;
        let bf16 = std::env::var("LIA_MINICPM_VISION_BF16").as_deref() == Ok("1");
        self.minicpmo_resample(image.width / 14, &encoded, bf16)
    }

    fn minicpmo_siglip(&self, image: &PreparedImage) -> Result<Buffer> {
        anyhow::ensure!(
            self.vision_eps.is_some(),
            "no MiniCPM-o vision encoder loaded"
        );
        anyhow::ensure!(
            image.width > 0
                && image.height > 0
                && image.width.is_multiple_of(14)
                && image.height.is_multiple_of(14)
                && image.pixels.len() == image.width * image.height * 3
                && image.pixels.iter().all(|v| v.is_finite()),
            "invalid MiniCPM-o RGB planes"
        );
        let width = image.width / 14;
        let height = image.height / 14;
        let patches = width * height;
        anyhow::ensure!(
            patches <= crate::gpu::MAX_PREFILL_BATCH,
            "MiniCPM-o image exceeds encoder capacity"
        );
        let bf16 = std::env::var("LIA_MINICPM_VISION_BF16").as_deref() == Ok("1");
        let pixels = DeviceBuffer::from_host(&self.stream, &image.pixels)?;
        let mut input = self.zeros(patches * 588)?;
        // SAFETY: complete RGB planes, positive 14-pixel grids and matching patch output.
        unsafe {
            self.module.minicpm_vision.minicpm_patches(
                &self.stream,
                LaunchConfig::for_num_elems(input.len() as u32),
                image.width,
                image.height,
                &pixels,
                &mut input,
            )?;
        }
        let input = self.mini_round(input, bf16)?;
        let embedded = self.vision_mm("v.patch_embd.weight", &input, patches)?;
        drop(input);
        let mut hidden = self.zeros(patches * 1152)?;
        // SAFETY: the loader checked the learned 70x70 table and 1152-channel bias.
        unsafe {
            if bf16 {
                self.module.minicpm_vision.minicpm_position::<true>(
                    &self.stream,
                    LaunchConfig::for_num_elems(hidden.len() as u32),
                    width,
                    height,
                    &embedded,
                    self.floats("vision.v.patch_embd.bias")?,
                    self.floats("vision.v.position_embd.weight")?,
                    &mut hidden,
                )?;
            } else {
                self.module.minicpm_vision.minicpm_position::<false>(
                    &self.stream,
                    LaunchConfig::for_num_elems(hidden.len() as u32),
                    width,
                    height,
                    &embedded,
                    self.floats("vision.v.patch_embd.bias")?,
                    self.floats("vision.v.position_embd.weight")?,
                    &mut hidden,
                )?;
            }
        }
        drop(embedded);
        for layer in 0..27 {
            hidden = self.mini_block(&format!("v.blk.{layer}"), &hidden, patches, bf16)?;
        }
        self.mini_norm("v.post_ln", &hidden, bf16)
    }

    fn minicpmo_resample(
        &self,
        width: usize,
        input: &DeviceBuffer<f32>,
        bf16: bool,
    ) -> Result<Buffer> {
        let patches = input.len() / 1152;
        let projected = self.vision_mm("r.kv.weight", input, patches)?;
        let projected = self.mini_round(projected, bf16)?;
        let normalized = self.mini_norm("r.ln_kv", &projected, bf16)?;
        let positioned = self.zeros(normalized.len())?;
        // SAFETY: normalized rows contain 4096 floats in a positive raster grid.
        unsafe {
            if bf16 {
                self.module.resampler.resampler_positions::<true>(
                    &self.stream,
                    LaunchConfig::for_num_elems(positioned.len() as u32),
                    width,
                    &normalized,
                    positioned.cu_deviceptr() as *mut f32,
                )?;
            } else {
                self.module.resampler.resampler_positions::<false>(
                    &self.stream,
                    LaunchConfig::for_num_elems(positioned.len() as u32),
                    width,
                    &normalized,
                    positioned.cu_deviceptr() as *mut f32,
                )?;
            }
        }
        let query = self.mini_norm("r.ln_q", self.floats("vision.r.query")?, bf16)?;
        let q = self.vision_mm("r.q.weight", &query, 64)?;
        let q = self.mini_affine(q, "r.q.bias", false, bf16)?;
        let k = self.vision_mm("r.k.weight", &positioned, patches)?;
        let k = self.mini_affine(k, "r.k.bias", false, bf16)?;
        let v = self.vision_mm("r.v.weight", &normalized, patches)?;
        let v = self.mini_affine(v, "r.v.bias", false, bf16)?;
        anyhow::ensure!(
            q.len() == 64 * 4096 && k.len() == patches * 4096 && v.len() == k.len(),
            "invalid resampler projections"
        );
        let attended = self.zeros(64 * 4096)?;
        // SAFETY: complete 32-head, 128-value rows and separate query/key counts.
        unsafe {
            self.module.resampler.resampler_attention(
                &self.stream,
                LaunchConfig::for_num_elems(64 * 32 * 32),
                64,
                patches,
                &q,
                &k,
                &v,
                attended.cu_deviceptr() as *mut f32,
            )?;
        }
        let attended = self.mini_round(attended, bf16)?;
        let output = self.vision_mm("r.out.weight", &attended, 64)?;
        let output = self.mini_affine(output, "r.out.bias", false, bf16)?;
        let normalized = self.mini_norm("r.ln_post", &output, bf16)?;
        let output = self.vision_mm("r.proj.weight", &normalized, 64)?;
        self.mini_round(output, bf16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context, ensure};

    #[test]
    #[ignore = "requires MiniCPM-o AWQ checkpoint, official vision oracle and CUDA"]
    fn siglip_matches_official_reference() -> Result<()> {
        let model = std::env::var("LIA_MINICPMO45_MODEL")?;
        let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_VISION_ORACLE")?);
        let info: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
        let bytes = std::fs::read(info["image"].as_str().context("missing image")?)?;
        let prepared = crate::models::minicpm_o45::prepare(&bytes, 1)?;
        ensure!(prepared.slices.len() == 1, "expected one reference slice");
        let engine = Engine::load_huggingface(&model, 128, true)?;
        let encoded = engine.minicpmo_siglip(&prepared.slices[0])?;
        let actual = encoded.to_host_vec(&engine.stream)?;
        let expected = std::fs::read(root.join("siglip.f32"))?;
        ensure!(
            expected.len() == actual.len() * 4,
            "invalid SigLIP reference extent"
        );
        let mut maximum = 0.0f32;
        let mut squared = 0.0f64;
        for (value, bytes) in actual.iter().zip(expected.chunks_exact(4)) {
            let reference = f32::from_le_bytes(bytes.try_into().unwrap());
            ensure!(
                value.is_finite() && reference.is_finite(),
                "non-finite SigLIP feature"
            );
            let error = value - reference;
            maximum = maximum.max(error.abs());
            squared += f64::from(error).powi(2);
        }
        let rms = (squared / actual.len() as f64).sqrt();
        println!("MiniCPM-o SigLIP features: max_abs={maximum}, rms={rms}");
        std::fs::write(
            root.join("native-siglip.f32"),
            actual
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        ensure!(
            maximum < 0.05 && rms < 0.01,
            "independent SigLIP feature mismatch"
        );
        Ok(())
    }
}
