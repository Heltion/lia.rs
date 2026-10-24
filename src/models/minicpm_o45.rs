//! Contracts for the official MiniCPM-o 4.5 AWQ checkpoint.
//!
//! This validates storage and dimensions; it does not advertise a serving path.
pub mod audio;
#[cfg(feature = "cuda")]
pub(crate) mod audio_weights;
#[cfg(feature = "cuda")]
pub(crate) mod cfm_weights;
#[cfg(feature = "cuda")]
pub(crate) mod codec_weights;
pub mod duplex;
pub mod flow_mel;
pub mod images;
pub mod kaldi;
mod pixels;
pub mod reference;
pub mod resample;
#[cfg(feature = "cuda")]
pub(crate) mod s3_weights;
#[cfg(feature = "cuda")]
pub(crate) mod speaker;
#[cfg(any(feature = "cuda", test))]
pub(crate) mod speech_sampling;
pub mod speech_text;
#[cfg(feature = "cuda")]
pub(crate) mod speech_weights;
pub(crate) mod stream_layout;
mod voice;
pub mod wav;
use crate::safetensors::{Archive, Dtype};
use anyhow::{Context, Result, ensure};
pub use pixels::{Prepared, prepare};
pub use voice::{MAX_CODEC_CODES, MAX_MEL_FRAMES, PreparedVoice, Voice};
mod audio_prompt;
mod prompt;
pub use prompt::{PreparedPrompt, prepare_media_prompt, prepare_prompt};
#[cfg(feature = "cuda")]
pub(crate) mod vision;
use serde_json::Value;
use std::{fs::File, path::Path};

pub(super) fn checkpoint(config: Value) -> Result<super::Checkpoint> {
    let parameters = Config::parse(&config)?;
    Ok(super::Checkpoint {
        architecture: super::Architecture::MiniCpmO45,
        vocabulary: parameters.vocabulary,
        context_limit: parameters.context_limit,
        config,
        quantization: super::Quantization::Awq,
        rms_epsilon: 1e-6,
        rotary_base: 1_000_000.0,
    })
}

pub struct Config {
    pub context_limit: usize,
    pub vocabulary: usize,
    pub hidden: usize,
    pub layers: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dimension: usize,
    pub group_size: usize,
}

fn dimensions(config: &Value, component: &str, fields: &[(&str, u64)]) -> Result<()> {
    for &(name, expected) in fields {
        ensure!(
            config[name].as_u64() == Some(expected),
            "unsupported MiniCPM-o 4.5 {component} {name}"
        );
    }
    Ok(())
}

impl Config {
    pub fn open(directory: impl AsRef<Path>) -> Result<Self> {
        let path = directory.as_ref().join("config.json");
        let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
        let config = serde_json::from_reader(file).context("reading MiniCPM-o config")?;
        Self::parse(&config).with_context(|| format!("validating {}", path.display()))
    }

    pub fn parse(config: &Value) -> Result<Self> {
        ensure!(
            config["model_type"] == "minicpmo"
                && config["version"] == "4.5"
                && config["architectures"] == serde_json::json!(["MiniCPMO"]),
            "expected official MiniCPM-o 4.5 architecture"
        );
        dimensions(
            config,
            "text",
            &[
                ("hidden_size", 4096),
                ("intermediate_size", 12288),
                ("num_hidden_layers", 36),
                ("num_attention_heads", 32),
                ("num_key_value_heads", 8),
                ("head_dim", 128),
                ("vocab_size", 151748),
                ("max_position_embeddings", 40960),
                ("query_num", 64),
                ("audio_pool_step", 5),
            ],
        )?;
        ensure!(
            config["attention_bias"] == false
                && config["hidden_act"] == "silu"
                && config["tie_word_embeddings"] == false
                && config["rope_scaling"].is_null()
                && config["rope_theta"] == 1_000_000
                && config["rms_norm_eps"] == 1e-6,
            "unsupported MiniCPM-o text normalization, rotary or projection layout"
        );
        ensure!(
            config["init_vision"] == true
                && config["init_audio"] == true
                && config["init_tts"] == true,
            "MiniCPM-o checkpoint requires all modality components"
        );
        let quant = &config["quantization_config"];
        ensure!(
            quant["quant_method"] == "awq"
                && quant["version"] == "gemm"
                && quant["bits"] == 4
                && quant["group_size"] == 128
                && quant["zero_point"] == true,
            "unsupported MiniCPM-o AWQ storage"
        );
        dimensions(
            &config["vision_config"],
            "vision",
            &[
                ("hidden_size", 1152),
                ("intermediate_size", 4304),
                ("num_hidden_layers", 27),
                ("num_attention_heads", 16),
                ("patch_size", 14),
            ],
        )?;
        dimensions(
            &config["audio_config"],
            "audio",
            &[
                ("d_model", 1024),
                ("encoder_layers", 24),
                ("encoder_attention_heads", 16),
                ("encoder_ffn_dim", 4096),
                ("num_mel_bins", 80),
                ("max_source_positions", 1500),
            ],
        )?;
        dimensions(
            &config["tts_config"],
            "speech",
            &[
                ("hidden_size", 768),
                ("intermediate_size", 3072),
                ("num_hidden_layers", 20),
                ("num_attention_heads", 12),
                ("num_key_value_heads", 12),
                ("num_audio_tokens", 6562),
                ("num_text_tokens", 152064),
            ],
        )?;
        Ok(Self {
            context_limit: 40960,
            vocabulary: 151748,
            hidden: 4096,
            layers: 36,
            query_heads: 32,
            kv_heads: 8,
            head_dimension: 128,
            group_size: 128,
        })
    }

    /// Validate text tensor metadata without allocating or unpacking model weights.
    pub fn validate_text_weights(&self, archive: &Archive) -> Result<()> {
        let check = |name: &str, dtype: Dtype, shape: &[usize]| -> Result<()> {
            let tensor = archive.tensor(name)?;
            ensure!(
                tensor.dtype == dtype && tensor.shape == shape,
                "MiniCPM-o tensor {name} has incompatible storage or dimensions"
            );
            Ok(())
        };
        for name in ["llm.model.embed_tokens.weight", "llm.lm_head.weight"] {
            check(name, Dtype::Bf16, &[self.vocabulary, self.hidden])?;
        }
        check("llm.model.norm.weight", Dtype::Bf16, &[self.hidden])?;
        for layer in 0..self.layers {
            let prefix = format!("llm.model.layers.{layer}");
            for name in ["input_layernorm", "post_attention_layernorm"] {
                check(
                    &format!("{prefix}.{name}.weight"),
                    Dtype::Bf16,
                    &[self.hidden],
                )?;
            }
            for name in ["q_norm", "k_norm"] {
                check(
                    &format!("{prefix}.self_attn.{name}.weight"),
                    Dtype::Bf16,
                    &[self.head_dimension],
                )?;
            }
            for (name, rows, columns) in [
                ("self_attn.q_proj", 4096, 4096),
                ("self_attn.k_proj", 1024, 4096),
                ("self_attn.v_proj", 1024, 4096),
                ("self_attn.o_proj", 4096, 4096),
                ("mlp.gate_proj", 12288, 4096),
                ("mlp.up_proj", 12288, 4096),
                ("mlp.down_proj", 4096, 12288),
            ] {
                let name = format!("{prefix}.{name}");
                check(&format!("{name}.qweight"), Dtype::I32, &[columns, rows / 8])?;
                check(
                    &format!("{name}.qzeros"),
                    Dtype::I32,
                    &[columns / self.group_size, rows / 8],
                )?;
                check(
                    &format!("{name}.scales"),
                    Dtype::F16,
                    &[columns / self.group_size, rows],
                )?;
            }
        }
        Ok(())
    }
}
