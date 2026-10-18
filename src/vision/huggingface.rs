use super::*;
use std::{fs::File, path::Path};

impl Processor {
    pub fn from_huggingface(
        directory: impl AsRef<Path>,
        min_tokens: usize,
        max_tokens: usize,
    ) -> Result<Self> {
        let path = directory.as_ref().join("preprocessor_config.json");
        let config: serde_json::Value = serde_json::from_reader(
            File::open(&path).with_context(|| format!("opening {}", path.display()))?,
        )?;
        ensure!(
            config["patch_size"] == 16
                && config["temporal_patch_size"] == 2
                && config["merge_size"] == 2,
            "unsupported Hugging Face image patch configuration"
        );
        ensure!(
            min_tokens > 0 && min_tokens <= max_tokens && max_tokens <= 16384,
            "invalid image token limits"
        );
        let array = |name: &str| -> Result<[f32; 3]> {
            let values = config[name]
                .as_array()
                .with_context(|| format!("missing image {name}"))?;
            ensure!(values.len() == 3, "image {name} must have three channels");
            let mut result = [0.0; 3];
            for (dst, value) in result.iter_mut().zip(values) {
                *dst = value
                    .as_f64()
                    .context("invalid image normalization value")? as f32;
                ensure!(dst.is_finite(), "nonfinite image normalization value");
            }
            Ok(result)
        };
        let mean = array("image_mean")?;
        let std = array("image_std")?;
        ensure!(
            std.iter().all(|x| *x > 0.0),
            "invalid image standard deviations"
        );
        Ok(Self {
            mean,
            std,
            min_pixels: min_tokens * 1024,
            max_pixels: max_tokens * 1024,
        })
    }
}
