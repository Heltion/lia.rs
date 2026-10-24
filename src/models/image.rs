//! Image preprocessing selection follows checkpoint architecture.
use super::{Architecture, Checkpoint};
use anyhow::{Context, Result, ensure};
use std::path::Path;

pub(crate) enum Processor {
    Qwen(crate::vision::Processor),
    MiniCpmV47 { max_slices: usize },
    MiniCpmO45 { max_slices: usize },
}
impl Processor {
    pub fn from_huggingface(directory: &str, minimum: usize, maximum: usize) -> Result<Self> {
        let checkpoint = Checkpoint::open(directory)?;
        match checkpoint.architecture {
            Architecture::MiniCpm5 => anyhow::bail!("MiniCPM5 is a text-only checkpoint"),
            Architecture::MiniCpmO45 => {
                let path = Path::new(directory).join("preprocessor_config.json");
                let value: serde_json::Value = serde_json::from_reader(
                    std::fs::File::open(&path)
                        .with_context(|| format!("opening {}", path.display()))?,
                )?;
                ensure!(
                    value["patch_size"] == 14
                        && value["scale_resolution"] == 448
                        && value["image_feature_size"] == 64
                        && value["slice_mode"] == true
                        && value["use_image_id"] == true
                        && value["norm_mean"] == serde_json::json!([0.5, 0.5, 0.5])
                        && value["norm_std"] == serde_json::json!([0.5, 0.5, 0.5]),
                    "unsupported MiniCPM-o image preprocessing"
                );
                let max_slices = usize::try_from(
                    value["max_slice_nums"]
                        .as_u64()
                        .context("missing image slice limit")?,
                )?;
                ensure!((1..=36).contains(&max_slices), "invalid image slice limit");
                Ok(Self::MiniCpmO45 { max_slices })
            }
            Architecture::Qwen35Moe => Ok(Self::Qwen(crate::vision::Processor::from_huggingface(
                directory, minimum, maximum,
            )?)),
            Architecture::MiniCpmV47 => {
                let path = Path::new(directory).join("processor_config.json");
                let value: serde_json::Value = serde_json::from_reader(
                    std::fs::File::open(&path)
                        .with_context(|| format!("opening {}", path.display()))?,
                )?;
                let image = &value["image_processor"];
                ensure!(
                    image["patch_size"] == 14
                        && image["scale_resolution"] == 448
                        && image["image_mean"] == serde_json::json!([0.5, 0.5, 0.5])
                        && image["image_std"] == serde_json::json!([0.5, 0.5, 0.5])
                        && image["downsample_mode"] == "16x"
                        && image["use_image_id"] == true
                        && image["slice_mode"] == true,
                    "unsupported MiniCPM image preprocessing"
                );
                let max_slices = usize::try_from(
                    image["max_slice_nums"]
                        .as_u64()
                        .context("missing MiniCPM slice limit")?,
                )?;
                ensure!(
                    (1..=36).contains(&max_slices),
                    "invalid MiniCPM slice limit"
                );
                Ok(Self::MiniCpmV47 { max_slices })
            }
        }
    }
}
