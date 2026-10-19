use super::*;
use anyhow::Context;

impl Engine {
    pub fn load(path: &str, capacity: usize) -> Result<Self> {
        let model = Model::open(path).context("reading model metadata")?;
        Self::load_from_model(path, capacity, &model)
    }
    pub fn load_from_model(path: &str, capacity: usize, model: &Model) -> Result<Self> {
        Self::load_models(path, capacity, model, None)
    }
    pub fn load_with_vision(path: &str, capacity: usize, projector: &str) -> Result<Self> {
        let model = Model::open(path).context("reading model metadata")?;
        Self::load_models(path, capacity, &model, Some(projector))
    }
    fn load_models(
        path: &str,
        capacity: usize,
        model: &Model,
        projector: Option<&str>,
    ) -> Result<Self> {
        if !matches!(model.metadata.get("general.architecture"), Some(Value::String(s)) if s == "qwen35")
        {
            return Err(anyhow!("only qwen35 is implemented"));
        }
        validate_capacity(
            capacity,
            usize::try_from(config_u64(model, "qwen35.context_length")?)?,
        )?;
        for (key, expected) in [
            ("block_count", 65),
            ("nextn_predict_layers", 1),
            ("embedding_length", 5120),
            ("feed_forward_length", 17408),
            ("attention.head_count", 24),
            ("attention.head_count_kv", 4),
            ("attention.key_length", 256),
            ("attention.value_length", 256),
            ("ssm.conv_kernel", 4),
            ("ssm.inner_size", 6144),
            ("ssm.state_size", 128),
            ("ssm.time_step_rank", 48),
            ("ssm.group_count", 16),
            ("full_attention_interval", 4),
            ("rope.dimension_count", 64),
        ] {
            if config_u64(model, &format!("qwen35.{key}"))? != expected {
                return Err(anyhow!("unsupported qwen35 dimension {key}"));
            }
        }
        validate_weights(model)?;
        anyhow::ensure!(
            matches!(model.metadata.get("qwen35.rope.dimension_sections"),
            Some(Value::Array { values, .. }) if values.len()==4 && values.iter().zip([11,11,10,0])
                .all(|(v,n)| matches!(v,Value::Unsigned(x) if *x==n)
                    || matches!(v,Value::Signed(x) if *x==n as i64))),
            "unsupported interleaved rotary sections"
        );
        let eps = config_f32(model, "qwen35.attention.layer_norm_rms_epsilon")?;
        let rope_base = config_f32(model, "qwen35.rope.freq_base")?;
        if !eps.is_finite() || eps <= 0.0 || !rope_base.is_finite() || rope_base <= 0.0 {
            return Err(anyhow!("invalid normalization or RoPE parameters"));
        }
        let context = CudaContext::new(0)?;
        let cuda_graph = std::env::var("LIA_CUDA_GRAPH").as_deref() == Ok("1");
        anyhow::ensure!(
            !cuda_graph || std::env::var_os("LIA_PROFILE").is_none(),
            "CUDA Graph cannot capture synchronizing LIA_PROFILE operations"
        );
        let stream = if cuda_graph {
            context.new_stream()?
        } else {
            context.default_stream()
        };
        let module = kernels::load(&context).context("loading CUDA operator modules")?;
        let mut file = File::open(path)?;
        let mut weights = BTreeMap::new();
        let fp8_prefill = std::env::var("LIA_PREFILL_FP8").as_deref() == Ok("1");
        anyhow::ensure!(
            fp8_prefill || std::env::var("LIA_PREFILL_WEIGHT_PREPACK").as_deref() != Ok("1"),
            "FP8 weight prepacking requires LIA_PREFILL_FP8=1"
        );
        let placement = host_weights::Placement::from_env()?;
        let mut host_weight_bytes = 0;
        for tensor in &model.tensors {
            let mut host = vec![0; tensor.byte_len as usize];
            file.seek(SeekFrom::Start(model.data_offset + tensor.offset))?;
            file.read_exact(&mut host)
                .with_context(|| format!("reading tensor {}", tensor.name))?;
            let (block, bytes) = crate::gguf::layout(tensor.kind)
                .ok_or_else(|| anyhow!("unsupported storage layout"))?;
            let on_host = placement.contains(&tensor.name);
            let floats = if tensor.kind == 0 {
                let values: Vec<f32> = host
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                Some(if on_host {
                    host_weights::allocate(&context, &values)?
                } else {
                    DeviceBuffer::from_host(&stream, &values)?
                })
            } else {
                None
            };
            let data = if on_host {
                host_weight_bytes += host.len() + floats.as_ref().map_or(0, |v| v.num_bytes());
                host_weights::allocate(&context, &host)?
            } else {
                DeviceBuffer::from_host(&stream, &host)?
            };
            weights.insert(
                tensor.name.clone(),
                Weight {
                    group: None,
                    data,
                    floats,
                    shape: tensor.shape.iter().map(|x| *x as usize).collect(),
                    kind: tensor.kind,
                    fp8_scale: if fp8_prefill
                        && tensor.shape.len() == 2
                        && tensor.kind != 0
                        && tensor.kind != 30
                    {
                        Some(
                            crate::quant::fp8::normalizer(tensor.kind, &host).with_context(
                                || format!("normalizing FP8 weights {}", tensor.name),
                            )?,
                        )
                    } else {
                        None
                    },
                    block: block as usize,
                    bytes: bytes as usize,
                },
            );
        }
        let mut engine = Self::initialize(
            context,
            stream,
            module,
            weights,
            capacity,
            Geometry::dense(),
            eps,
            rope_base,
            fp8_prefill,
            cuda_graph,
        )?;
        engine.host_weight_bytes = host_weight_bytes;
        if let Some(path) = projector {
            if vision::offload::enabled() {
                let cache = engine.park_attention()?;
                engine.load_vision(path)?;
                engine.host_vision =
                    vision::offload::park_weights(&mut engine.weights, &engine.stream)?;
                engine.restore_attention(cache)?;
            } else {
                engine.load_vision(path)?;
            }
        }
        engine.prepare_buffers()?;
        Ok(engine)
    }
}

fn validate_capacity(capacity: usize, model_limit: usize) -> Result<()> {
    let limit = model_limit.min(crate::MAX_CONTEXT_TOKENS);
    anyhow::ensure!(
        capacity > 0 && capacity <= limit,
        "context capacity must be 1..={limit}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_capacity_respects_model_and_index_limits() -> Result<()> {
        for capacity in [1, 32768, 49152, 65536, 131072, 262144] {
            validate_capacity(capacity, 262144)?;
        }
        assert!(validate_capacity(0, 262144).is_err());
        assert!(validate_capacity(65536, 32768).is_err());
        assert!(validate_capacity(262145, usize::MAX).is_err());
        assert!(validate_capacity(1, 0).is_err());
        Ok(())
    }
}
