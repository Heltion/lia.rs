//! Native sharded GSQ checkpoint loading; inference uses the Rust CUDA path.
use super::*;
use crate::models::qwen35::layout;
use crate::{
    quant::gsq::Matrix,
    safetensors::{Archive, Dtype},
};
use anyhow::{Context, ensure};
mod minicpm5;
mod minicpmo;
mod minicpmo_audio;
mod minicpmo_speech;
mod parameters;
mod vision;
mod weights;

struct Loader<'a> {
    archive: &'a Archive,
    architecture: crate::models::Architecture,
    expert_bits: u32,
    stream: &'a Arc<CudaStream>,
}

impl Engine {
    pub fn load_huggingface(directory: &str, capacity: usize, vision: bool) -> Result<Self> {
        let checkpoint = crate::models::Checkpoint::open(directory)?;
        if checkpoint.architecture == crate::models::Architecture::MiniCpm5 {
            return Self::load_minicpm5(directory, capacity, &checkpoint);
        }
        if checkpoint.architecture == crate::models::Architecture::MiniCpmO45 {
            return Self::load_minicpmo(directory, capacity, &checkpoint, vision);
        }
        let config = &checkpoint.config;
        checkpoint.validate_capacity(capacity)?;
        let eps = checkpoint.rms_epsilon;
        let rope = checkpoint.rotary_base;
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
            architecture: checkpoint.architecture,
            expert_bits: if checkpoint.quantization == crate::models::Quantization::Int4 {
                4
            } else {
                2
            },
            stream: &stream,
        };
        let mut weights = BTreeMap::new();
        weights.insert(
            "token_embd.weight".into(),
            loader.dense(
                "model.language_model.embed_tokens.weight",
                &[checkpoint.vocabulary, 2048],
            )?,
        );
        weights.insert(
            "output_norm.weight".into(),
            loader.floats("model.language_model.norm.weight", &[2048], |x| x + 1.0)?,
        );
        weights.insert(
            "output.weight".into(),
            if checkpoint.architecture == crate::models::Architecture::MiniCpmV47 {
                loader.dense("lm_head.weight", &[checkpoint.vocabulary, 2048])?
            } else {
                loader.quant("lm_head", 8, 0, checkpoint.vocabulary, 2048)?
            },
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
            if checkpoint.architecture == crate::models::Architecture::MiniCpmV47 {
                loader.parameters(
                    crate::models::minicpm_v47::weights::plan(config)?,
                    &mut weights,
                )?;
            } else {
                loader.vision(&config["vision_config"], &mut weights)?;
            }
        }
        let geometry = Geometry {
            llama: false,
            qwen3: false,
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
        if checkpoint.architecture == crate::models::Architecture::MiniCpmV47 {
            engine.vision_encoder = crate::models::VisionEncoder::MiniCpmV47;
        }
        engine.host_weight_bytes = host_embedding_bytes;
        engine.experts = experts;
        if std::env::var("LIA_MOE_STAGE_HOST").as_deref() == Ok("1") {
            let experts = engine
                .experts
                .values()
                .next()
                .context("missing expert staging geometry")?;
            engine
                .routing
                .as_mut()
                .context("missing routing workspace")?
                .grouped
                .reserve_host_stage(&engine.stream, experts)?;
        }
        if vision {
            engine.vision_eps = Some(1e-6);
        }
        engine.prepare_buffers()?;
        Ok(engine)
    }
}
