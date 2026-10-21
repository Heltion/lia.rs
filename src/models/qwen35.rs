//! Supported geometry shared by the Qwen 3.5 MoE language backbones.
use anyhow::{Context, Result, ensure};
use serde_json::Value;
#[cfg(any(feature = "cuda", test))]
pub(crate) mod layout;
#[cfg(any(feature = "cuda", test))]
pub(crate) mod weights;

pub(super) fn validate(text: &Value, vocabulary: u64) -> Result<()> {
    let types = text["layer_types"]
        .as_array()
        .context("missing layer types")?;
    ensure!(
        types.len() == 40
            && types.iter().enumerate().all(|(i, t)| t
                == if (i + 1) % 4 == 0 {
                    "full_attention"
                } else {
                    "linear_attention"
                }),
        "unsupported Qwen MoE layer ordering"
    );
    for (name, expected) in [
        ("hidden_size", 2048),
        ("num_hidden_layers", 40),
        ("num_attention_heads", 16),
        ("num_key_value_heads", 2),
        ("head_dim", 256),
        ("full_attention_interval", 4),
        ("linear_num_key_heads", 16),
        ("linear_num_value_heads", 32),
        ("linear_key_head_dim", 128),
        ("linear_value_head_dim", 128),
        ("linear_conv_kernel_dim", 4),
        ("num_experts", 256),
        ("num_experts_per_tok", 8),
        ("moe_intermediate_size", 512),
        ("shared_expert_intermediate_size", 512),
        ("vocab_size", vocabulary),
    ] {
        ensure!(
            text[name].as_u64() == Some(expected),
            "unsupported Qwen MoE dimension {name}"
        );
    }
    ensure!(
        text["hidden_act"] == "silu"
            && text["attention_bias"] == false
            && text["attn_output_gate"] == true,
        "unsupported Qwen MoE activation or attention configuration"
    );
    ensure!(
        text["rope_parameters"]["mrope_interleaved"] == true
            && text["rope_parameters"]["mrope_section"] == serde_json::json!([11, 11, 10])
            && text["rope_parameters"]["partial_rotary_factor"] == 0.25,
        "unsupported Qwen MoE rotary layout"
    );
    Ok(())
}
