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
        if capacity == 0 || capacity > 32768 {
            return Err(anyhow!("context capacity must be 1..=32768"));
        }
        if !matches!(model.metadata.get("general.architecture"), Some(Value::String(s)) if s == "qwen35")
        {
            return Err(anyhow!("only qwen35 is implemented"));
        }
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
        for tensor in &model.tensors {
            let mut host = vec![0; tensor.byte_len as usize];
            file.seek(SeekFrom::Start(model.data_offset + tensor.offset))?;
            file.read_exact(&mut host)
                .with_context(|| format!("reading tensor {}", tensor.name))?;
            let (block, bytes) = crate::gguf::layout(tensor.kind)
                .ok_or_else(|| anyhow!("unsupported storage layout"))?;
            let floats = if tensor.kind == 0 {
                let values: Vec<f32> = host
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                Some(DeviceBuffer::from_host(&stream, &values)?)
            } else {
                None
            };
            let data = DeviceBuffer::from_host(&stream, &host)?;
            weights.insert(
                tensor.name.clone(),
                Weight {
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
        let mut states = Vec::new();
        for layer in 0..65 {
            states.push(if layer == 64 || (layer + 1) % 4 == 0 {
                State::Attention {
                    k: DeviceBuffer::zeroed(&stream, capacity * 1024)?,
                    v: DeviceBuffer::zeroed(&stream, capacity * 1024)?,
                }
            } else {
                State::Recurrent {
                    conv: DeviceBuffer::zeroed(&stream, 3 * 10240)?,
                    matrix: DeviceBuffer::zeroed(&stream, 48 * 128 * 128)?,
                }
            });
        }
        stream.synchronize()?;
        let rotary = RefCell::new(RotaryWork {
            data: DeviceBuffer::zeroed(&stream, crate::gpu::MAX_PREFILL_BATCH * 64)?,
            key: None,
            positions: DeviceBuffer::zeroed(&stream, capacity * 4)?,
            plan: None,
        });
        let token_ids = DeviceBuffer::zeroed(&stream, crate::gpu::MAX_PREFILL_BATCH)?;
        let zero_hidden = DeviceBuffer::zeroed(&stream, 5120)?;
        let mut engine = Self {
            _context: context,
            stream,
            module,
            weights,
            states,
            position: 0,
            capacity,
            eps,
            rope_base,
            pool: Rc::new(RefCell::new(BTreeMap::new())),
            last_hidden: None,
            mtp_prepared: None,
            profile: std::env::var_os("LIA_PROFILE").is_some(),
            integer: std::env::var("LIA_INT8").as_deref().unwrap_or("1") != "0",
            split_k: std::env::var("LIA_SPLIT_K")
                .unwrap_or_else(|_| "1".into())
                .parse::<usize>()
                .ok()
                .filter(|n| *n == 1 || *n == 4)
                .ok_or_else(|| anyhow!("LIA_SPLIT_K must be 1 or 4"))?,
            row_group: std::env::var("LIA_MM_ROWS")
                .unwrap_or_else(|_| "2".into())
                .parse::<usize>()
                .ok()
                .filter(|n| *n == 1 || *n == 2)
                .ok_or_else(|| anyhow!("LIA_MM_ROWS must be 1 or 2"))?,
            matrix_threads: std::env::var("LIA_MM_THREADS")
                .unwrap_or_else(|_| "128".into())
                .parse::<usize>()
                .ok()
                .filter(|n| [64, 128].contains(n))
                .ok_or_else(|| anyhow!("LIA_MM_THREADS must be 64, 128 or 256"))?,
            attention_mma_decode: std::env::var("LIA_ATTN_MMA_DECODE").as_deref() == Ok("1"),
            attention_query_tiles: std::env::var("LIA_ATTN_QUERY_TILES").as_deref() == Ok("1"),
            attention_work: RefCell::new(None),
            prefill_batch: std::env::var("LIA_PREFILL_BATCH")
                .unwrap_or_else(|_| "512".into())
                .parse::<usize>()
                .ok()
                .filter(|batch| [512, 1024, 2048, 4096].contains(batch))
                .ok_or_else(|| anyhow!("LIA_PREFILL_BATCH must be 512, 1024, 2048, or 4096"))?,
            prefill_tile: std::env::var("LIA_PREFILL_TILE")
                .unwrap_or_else(|_| "128".into())
                .parse::<usize>()
                .ok()
                .filter(|tile| [64, 128].contains(tile))
                .ok_or_else(|| anyhow!("LIA_PREFILL_TILE must be 64 or 128"))?,
            fp8_work: RefCell::new(None),
            prefill_input: RefCell::new(None),
            preallocated: cuda_graph || std::env::var("LIA_PREALLOCATE").as_deref() == Ok("1"),
            cuda_graph,
            graphs: RefCell::new(BTreeMap::new()),
            graph_stats: RefCell::new([0; 3]),
            scratch_reserved: 0,
            token_ids,
            zero_hidden,
            gdn_prepared: None,
            greedy_work: BTreeMap::new(),
            quant_work: RefCell::new(BTreeMap::new()),
            sampling_work: RefCell::new(BTreeMap::new()),
            spec_states: Vec::new(),
            capture: false,
            max_draft: std::env::var("LIA_MTP_DRAFT_MAX")
                .unwrap_or_else(|_| "3".into())
                .parse::<usize>()
                .ok()
                .filter(|n| (1..=7).contains(n))
                .ok_or_else(|| anyhow!("LIA_MTP_DRAFT_MAX must be between 1 and 7"))?,
            stochastic_mtp: match std::env::var("LIA_MTP_SAMPLING")
                .as_deref()
                .unwrap_or("match")
            {
                "match" => false,
                "stochastic" => true,
                _ => return Err(anyhow!("LIA_MTP_SAMPLING must be match or stochastic")),
            },
            verify_hidden: None,
            rotary,
            vision_eps: None,
            images: Vec::new(),
        };
        engine.prepare_speculation_buffers()?;
        engine.prepare_small_workspaces()?;
        if std::env::var("LIA_GDN_PARALLEL").as_deref() == Ok("1") {
            engine.gdn_prepared = Some(DeviceBuffer::zeroed(
                &engine.stream,
                48 * crate::gpu::MAX_PREFILL_BATCH.div_ceil(16) * 1072,
            )?);
        }
        if fp8_prefill {
            let elements = 17408 * engine.prefill_batch.min(crate::gpu::MAX_PREFILL_BATCH);
            *engine.fp8_work.borrow_mut() = Some(Fp8Work {
                weight_tiles: if std::env::var("LIA_PREFILL_WEIGHT_PREPACK").as_deref() == Ok("1") {
                    Some(DeviceBuffer::zeroed(&engine.stream, 17408 * 5120 / 4)?)
                } else {
                    None
                },
                packed: DeviceBuffer::zeroed(&engine.stream, elements.div_ceil(4))?,
                maxima: DeviceBuffer::zeroed(&engine.stream, elements.div_ceil(1024))?,
                scales: DeviceBuffer::zeroed(&engine.stream, 2)?,
            });
        }
        if let Some(path) = projector {
            engine.load_vision(path)?;
        }
        engine.prepare_buffers()?;
        Ok(engine)
    }
}
