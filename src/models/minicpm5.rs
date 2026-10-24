//! Official MiniCPM5-2B BF16 Llama checkpoint contract.
use super::{Architecture, Checkpoint, Quantization};
use anyhow::{Result, ensure};
use serde_json::Value;

pub(super) fn checkpoint(config: Value) -> Result<Checkpoint> {
    for (name, expected) in [
        ("hidden_size", 2048),
        ("intermediate_size", 6144),
        ("num_hidden_layers", 42),
        ("num_attention_heads", 16),
        ("num_key_value_heads", 2),
        ("head_dim", 128),
        ("vocab_size", 130560),
        ("max_position_embeddings", 131072),
    ] {
        ensure!(
            config[name].as_u64() == Some(expected),
            "unsupported MiniCPM5 {name}"
        );
    }
    ensure!(
        config["architectures"] == serde_json::json!(["LlamaForCausalLM"])
            && config["hidden_act"] == "silu"
            && config["tie_word_embeddings"] == false
            && config["rope_scaling"].is_null()
            && config["quantization_config"].is_null()
            && config["rms_norm_eps"] == 1e-6
            && config["rope_theta"] == 5_000_000,
        "unsupported MiniCPM5 attention, normalization or checkpoint format"
    );
    ensure!(
        config["torch_dtype"] == "bfloat16" || config["dtype"] == "bfloat16",
        "MiniCPM5 requires BF16 checkpoint weights"
    );
    Ok(Checkpoint {
        architecture: Architecture::MiniCpm5,
        config,
        quantization: Quantization::Bf16,
        vocabulary: 130560,
        context_limit: 131072,
        rms_epsilon: 1e-6,
        rotary_base: 5_000_000.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_incompatible_llama_checkpoints() {
        assert!(checkpoint(serde_json::json!({"model_type":"llama"})).is_err());
    }
}
