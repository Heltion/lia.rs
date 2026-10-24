//! Native BF16 loading for the supported standard Llama checkpoint.
use super::*;

impl Engine {
    pub(super) fn load_minicpm5(
        directory: &str,
        capacity: usize,
        checkpoint: &crate::models::Checkpoint,
    ) -> Result<Self> {
        checkpoint.validate_capacity(capacity)?;
        let archive = Archive::open(directory)?;
        let context = CudaContext::new(0)?;
        let graph = std::env::var("LIA_CUDA_GRAPH").as_deref() == Ok("1");
        anyhow::ensure!(
            !graph || std::env::var_os("LIA_PROFILE").is_none(),
            "CUDA Graph cannot capture synchronizing profiling"
        );
        let stream = if graph {
            context.new_stream()?
        } else {
            context.default_stream()
        };
        let module = kernels::load(&context)?;
        let loader = Loader {
            archive: &archive,
            architecture: checkpoint.architecture,
            expert_bits: 0,
            stream: &stream,
        };
        let mut weights = BTreeMap::new();
        weights.insert(
            "token_embd.weight".into(),
            loader.dense("model.embed_tokens.weight", &[130560, 2048])?,
        );
        weights.insert(
            "output.weight".into(),
            loader.dense("lm_head.weight", &[130560, 2048])?,
        );
        weights.insert(
            "output_norm.weight".into(),
            loader.floats("model.norm.weight", &[2048], |v| v)?,
        );
        for layer in 0..42 {
            let target = format!("blk.{layer}");
            let source = format!("model.layers.{layer}");
            for (name, original) in [
                ("attn_norm", "input_layernorm"),
                ("post_attention_norm", "post_attention_layernorm"),
            ] {
                weights.insert(
                    format!("{target}.{name}.weight"),
                    loader.floats(&format!("{source}.{original}.weight"), &[2048], |v| v)?,
                );
            }
            for (name, original, rows, columns) in [
                ("attn_q", "self_attn.q_proj", 2048, 2048),
                ("attn_k", "self_attn.k_proj", 256, 2048),
                ("attn_v", "self_attn.v_proj", 256, 2048),
                ("attn_output", "self_attn.o_proj", 2048, 2048),
                ("ffn_gate", "mlp.gate_proj", 6144, 2048),
                ("ffn_up", "mlp.up_proj", 6144, 2048),
                ("ffn_down", "mlp.down_proj", 2048, 6144),
            ] {
                weights.insert(
                    format!("{target}.{name}.weight"),
                    loader.dense(&format!("{source}.{original}.weight"), &[rows, columns])?,
                );
            }
        }
        let geometry = Geometry {
            llama: true,
            qwen3: false,
            hidden: 2048,
            layers: 42,
            query_heads: 16,
            kv_heads: 2,
            value_heads: 0,
            mtp: false,
            moe: false,
        };
        let mut engine = Self::initialize(
            context,
            stream,
            module,
            weights,
            capacity,
            geometry,
            checkpoint.rms_epsilon,
            checkpoint.rotary_base,
            false,
            graph,
        )?;
        engine.prepare_buffers()?;
        engine.prepare_llama_weight_cache()?;
        Ok(engine)
    }
}
