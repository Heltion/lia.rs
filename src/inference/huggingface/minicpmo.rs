//! Native MiniCPM-o Qwen3 trunk loading; multimodal towers are loaded separately.
use super::*;

impl Engine {
    pub(super) fn load_minicpmo(
        directory: &str,
        capacity: usize,
        checkpoint: &crate::models::Checkpoint,
        vision: bool,
    ) -> Result<Self> {
        checkpoint.validate_capacity(capacity)?;
        let config = crate::models::minicpm_o45::Config::parse(&checkpoint.config)?;
        let archive = Archive::open(directory)?;
        config.validate_text_weights(&archive)?;
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
            expert_bits: 0,
            stream: &stream,
        };
        let mut weights = BTreeMap::new();
        weights.insert(
            "token_embd.weight".into(),
            loader.dense("llm.model.embed_tokens.weight", &[151748, 4096])?,
        );
        weights.insert(
            "output.weight".into(),
            loader.dense("llm.lm_head.weight", &[151748, 4096])?,
        );
        weights.insert(
            "output_norm.weight".into(),
            loader.floats("llm.model.norm.weight", &[4096], |v| v)?,
        );
        let host_bytes = if std::env::var("LIA_HOST_EMBEDDING").as_deref() == Ok("1") {
            weights["token_embd.weight"].data.len()
        } else {
            0
        };
        let register_gateup = match std::env::var("LIA_AWQ_GATEUP_REGISTER")
            .as_deref()
            .unwrap_or("0")
        {
            "0" => false,
            "1" => true,
            _ => anyhow::bail!("LIA_AWQ_GATEUP_REGISTER must be 0 or 1"),
        };
        let mut awq = BTreeMap::new();
        let fused_qkv = match std::env::var("LIA_AWQ_QKV").as_deref() {
            Ok("1") => true,
            Ok("0") | Err(_) => false,
            _ => anyhow::bail!("LIA_AWQ_QKV must be 0 or 1"),
        };
        let fused_gateup = match std::env::var("LIA_AWQ_GATEUP").as_deref() {
            Ok("1") => true,
            Ok("0") | Err(_) => false,
            _ => anyhow::bail!("LIA_AWQ_GATEUP must be 0 or 1"),
        };
        ensure!(
            !register_gateup || fused_gateup,
            "register gate/up requires LIA_AWQ_GATEUP=1"
        );
        for layer in 0..36 {
            let target = format!("blk.{layer}");
            let source = format!("llm.model.layers.{layer}");
            for (name, original, width) in [
                ("attn_norm", "input_layernorm", 4096),
                ("post_attention_norm", "post_attention_layernorm", 4096),
                ("attn_q_norm", "self_attn.q_norm", 128),
                ("attn_k_norm", "self_attn.k_norm", 128),
            ] {
                weights.insert(
                    format!("{target}.{name}.weight"),
                    loader.floats(&format!("{source}.{original}.weight"), &[width], |v| v)?,
                );
            }
            if fused_qkv {
                let matrices = ["q_proj", "k_proj", "v_proj"]
                    .map(|name| {
                        crate::quant::awq::Matrix::load(
                            &archive,
                            &format!("{source}.self_attn.{name}"),
                            128,
                        )
                    })
                    .into_iter()
                    .collect::<Result<Vec<_>>>()?;
                let matrix = crate::quant::awq::Matrix::concatenate(&matrices)?;
                awq.insert(
                    format!("{target}.attn_qkv.weight"),
                    crate::gpu::awq::Weights::upload(&stream, &matrix)?,
                );
            }
            if fused_gateup {
                let matrices = ["gate_proj", "up_proj"]
                    .map(|name| {
                        crate::quant::awq::Matrix::load(
                            &archive,
                            &format!("{source}.mlp.{name}"),
                            128,
                        )
                    })
                    .into_iter()
                    .collect::<Result<Vec<_>>>()?;
                let (suffix, matrix) = if register_gateup {
                    (
                        "ffn_gateup_interleaved.weight",
                        crate::quant::awq::Matrix::interleave_pair(&matrices, 8)?,
                    )
                } else {
                    (
                        "ffn_gateup.weight",
                        crate::quant::awq::Matrix::concatenate(&matrices)?,
                    )
                };
                awq.insert(
                    format!("{target}.{suffix}"),
                    crate::gpu::awq::Weights::upload(&stream, &matrix)?,
                );
            }
            for (name, original) in [
                ("attn_q", "self_attn.q_proj"),
                ("attn_k", "self_attn.k_proj"),
                ("attn_v", "self_attn.v_proj"),
                ("attn_output", "self_attn.o_proj"),
                ("ffn_gate", "mlp.gate_proj"),
                ("ffn_up", "mlp.up_proj"),
                ("ffn_down", "mlp.down_proj"),
            ] {
                if fused_qkv && ["attn_q", "attn_k", "attn_v"].contains(&name) {
                    continue;
                }
                if fused_gateup && ["ffn_gate", "ffn_up"].contains(&name) {
                    continue;
                }
                let matrix = crate::quant::awq::Matrix::load(
                    &archive,
                    &format!("{source}.{original}"),
                    128,
                )?;
                awq.insert(
                    format!("{target}.{name}.weight"),
                    crate::gpu::awq::Weights::upload(&stream, &matrix)?,
                );
            }
            eprintln!("loaded MiniCPM-o text block {}/36", layer + 1);
        }
        if vision {
            loader.parameters(crate::models::minicpm_o45::vision::plan(), &mut weights)?;
        }
        let geometry = Geometry {
            llama: false,
            qwen3: true,
            hidden: 4096,
            layers: 36,
            query_heads: 32,
            kv_heads: 8,
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
            cuda_graph,
        )?;
        engine.awq_weights = awq;
        engine.host_weight_bytes = host_bytes;
        if vision {
            engine.vision_eps = Some(1e-6);
            engine.vision_encoder = crate::models::VisionEncoder::MiniCpmO45;
        }
        engine.prepare_buffers()?;
        Ok(engine)
    }
}
