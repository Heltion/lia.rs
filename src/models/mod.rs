//! Model identity and checkpoint contracts, separate from execution resources.
pub mod minicpm_v47;
mod quantization;
pub(crate) mod qwen35;
pub use quantization::Quantization;

use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::{fs::File, path::Path};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Architecture {
    Qwen35Moe,
    MiniCpmV47,
}

pub struct Checkpoint {
    pub architecture: Architecture,
    pub config: Value,
    pub quantization: Quantization,
    pub vocabulary: usize,
    pub context_limit: usize,
    pub rms_epsilon: f32,
    pub rotary_base: f32,
}

impl Checkpoint {
    pub fn open(directory: impl AsRef<Path>) -> Result<Self> {
        let path = directory.as_ref().join("config.json");
        let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
        let config =
            serde_json::from_reader(file).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(config).with_context(|| format!("validating {}", path.display()))
    }

    pub fn parse(config: Value) -> Result<Self> {
        let architecture = match config["model_type"].as_str() {
            Some("qwen3_5_moe") => Architecture::Qwen35Moe,
            Some("minicpmv4_7") => Architecture::MiniCpmV47,
            other => anyhow::bail!("unsupported model architecture {other:?}"),
        };
        let text = &config["text_config"];
        let vocabulary = match architecture {
            Architecture::Qwen35Moe => 248320,
            Architecture::MiniCpmV47 => 248144,
        };
        qwen35::validate(text, vocabulary)?;
        let quantization = Quantization::parse(architecture, &config["quantization_config"])?;
        ensure!(
            config["tie_word_embeddings"] == false,
            "unsupported tied output embeddings"
        );
        let context_limit = usize::try_from(
            text["max_position_embeddings"]
                .as_u64()
                .context("missing model context limit")?,
        )
        .context("model context limit overflow")?;
        let rms_epsilon = text["rms_norm_eps"]
            .as_f64()
            .context("missing RMS epsilon")? as f32;
        let rotary_base = text["rope_parameters"]["rope_theta"]
            .as_f64()
            .context("missing rotary base")? as f32;
        ensure!(
            context_limit > 0
                && rms_epsilon.is_finite()
                && rms_epsilon > 0.0
                && rotary_base.is_finite()
                && rotary_base > 0.0,
            "invalid model numerical parameters"
        );
        Ok(Self {
            architecture,
            quantization,
            config,
            vocabulary: vocabulary as usize,
            context_limit,
            rms_epsilon,
            rotary_base,
        })
    }

    pub fn validate_capacity(&self, capacity: usize) -> Result<()> {
        ensure!(
            capacity > 0 && capacity <= crate::MAX_CONTEXT_TOKENS && capacity <= self.context_limit,
            "context capacity exceeds model or implementation limit"
        );
        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub(crate) mod vision;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisionEncoder {
    Qwen3Vl,
    MiniCpmV47,
}

#[cfg(feature = "cuda")]
pub(crate) mod image;
