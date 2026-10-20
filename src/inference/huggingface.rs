//! Native sharded GSQ checkpoint loading; inference uses the Rust CUDA path.
use super::*;
use crate::{
    quant::gsq::Matrix,
    safetensors::{Archive, Dtype},
};
use anyhow::{Context, ensure};
use std::path::Path;
mod layout;
mod vision;
mod weights;

struct Loader<'a> {
    archive: &'a Archive,
    stream: &'a Arc<CudaStream>,
}

impl Engine {
    pub fn load_huggingface(directory: &str, capacity: usize, vision: bool) -> Result<Self> {
        let config: serde_json::Value =
            serde_json::from_reader(File::open(Path::new(directory).join("config.json"))?)?;
        let text = &config["text_config"];
        ensure!(
            config["model_type"] == "qwen3_5_moe",
            "unsupported Hugging Face model architecture"
        );
        ensure!(
            config["tie_word_embeddings"] == false,
            "tied output embeddings are not implemented"
        );
        let quant = &config["quantization_config"];
        ensure!(
            quant["quant_method"] == "compressed-tensors" && quant["format"] == "pack-quantized",
            "unsupported GSQ checkpoint schema"
        );
        for (group, bits, strategy) in [
            ("group_0", 2, "group"),
            ("group_1", 8, "group"),
            ("group_2", 8, "channel"),
        ] {
            let weights = &quant["config_groups"][group]["weights"];
            ensure!(
                weights["num_bits"] == bits
                    && weights["symmetric"] == true
                    && weights["type"] == "int"
                    && weights["strategy"] == strategy
                    && (strategy == "channel" || weights["group_size"] == 128),
                "unsupported GSQ quantization group {group}"
            );
        }
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
            ("vocab_size", 248320),
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
        let limit = text["max_position_embeddings"]
            .as_u64()
            .context("missing model context limit")?;
        ensure!(
            capacity > 0 && capacity <= crate::MAX_CONTEXT_TOKENS && capacity as u64 <= limit,
            "context capacity exceeds the model or implementation limit"
        );
        let eps = text["rms_norm_eps"]
            .as_f64()
            .context("missing RMS epsilon")? as f32;
        let rope = text["rope_parameters"]["rope_theta"]
            .as_f64()
            .context("missing rotary base")? as f32;
        ensure!(
            eps.is_finite() && eps > 0.0 && rope.is_finite() && rope > 0.0,
            "invalid normalization or rotary parameters"
        );
        let archive = Archive::open(directory)?;
        let context = CudaContext::new(0)?;
        let cuda_graph = std::env::var("LIA_CUDA_GRAPH").as_deref() == Ok("1");
        ensure!(
            !cuda_graph || std::env::var_os("LIA_PROFILE").is_none(),
            "CUDA Graph cannot capture synchronizing profiling"
        );
        let stream = if cuda_graph {
            context.new_stream()?
        } else {
            context.default_stream()
        };
        let module = kernels::load(&context)?;
        let loader = Loader {
            archive: &archive,
            stream: &stream,
        };
        let mut weights = BTreeMap::new();
        weights.insert(
            "token_embd.weight".into(),
            loader.dense("model.language_model.embed_tokens.weight", &[248320, 2048])?,
        );
        weights.insert(
            "output_norm.weight".into(),
            loader.floats("model.language_model.norm.weight", &[2048], |x| x + 1.0)?,
        );
        weights.insert(
            "output.weight".into(),
            loader.quant("lm_head", 8, 0, 248320, 2048)?,
        );
        let offload = std::env::var("LIA_MOE_HOST_EXPERTS")
            .unwrap_or_else(|_| "0".into())
            .parse::<usize>()
            .context("invalid LIA_MOE_HOST_EXPERTS")?;
        ensure!(offload <= 256, "host expert count exceeds 256");
        eprintln!("MoE mapped-host experts per layer: {offload}/256");
        let mut experts = BTreeMap::new();
        for layer in 0..40 {
            loader.language_layer(layer, &mut weights)?;
            let prefix = format!("model.language_model.layers.{layer}.mlp.experts");
            let gate = loader.expert_stack(&prefix, "gate_proj", 512, 2048)?;
            let up = loader.expert_stack(&prefix, "up_proj", 512, 2048)?;
            let down = loader.expert_stack(&prefix, "down_proj", 2048, 512)?;
            let mut block = crate::gpu::moe::Experts::upload_offloaded(
                &stream, 2048, 512, gate, up, down, offload,
            )?;
            if std::env::var("LIA_MOE_REGISTER_MMA").as_deref() == Ok("1") {
                let enabled = block.prepare_register(&module, &stream)?;
                eprintln!("MoE register MMA block {layer}: {enabled}");
            }
            experts.insert(layer, block);
            eprintln!("loaded MoE block {}/40", layer + 1);
        }
        if vision {
            loader.vision(&config["vision_config"], &mut weights)?;
        }
        let geometry = Geometry {
            hidden: 2048,
            layers: 40,
            query_heads: 16,
            kv_heads: 2,
            value_heads: 32,
            mtp: false,
            moe: true,
        };
        let host_embedding_bytes = if std::env::var("LIA_HOST_EMBEDDING").as_deref() == Ok("1") {
            weights
                .get("token_embd.weight")
                .context("missing token embeddings")?
                .data
                .len()
        } else {
            0
        };
        let mut engine = Self::initialize(
            context, stream, module, weights, capacity, geometry, eps, rope, false, cuda_graph,
        )?;
        engine.host_weight_bytes = host_embedding_bytes;
        engine.experts = experts;
        if vision {
            engine.vision_eps = Some(1e-6);
        }
        engine.prepare_buffers()?;
        Ok(engine)
    }
}
