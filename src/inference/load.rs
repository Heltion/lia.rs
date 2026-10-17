use super::*;
use anyhow::Context;

impl Engine {
    pub fn load(path: &str, capacity: usize) -> Result<Self> {
        let model = Model::open(path).context("reading model metadata")?;
        Self::load_from_model(path, capacity, &model)
    }
    pub fn load_from_model(path: &str, capacity: usize, model: &Model) -> Result<Self> {
        if capacity == 0 || capacity > 16384 {
            return Err(anyhow!("context capacity must be 1..=16384"));
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
        let eps = config_f32(model, "qwen35.attention.layer_norm_rms_epsilon")?;
        let rope_base = config_f32(model, "qwen35.rope.freq_base")?;
        if !eps.is_finite() || eps <= 0.0 || !rope_base.is_finite() || rope_base <= 0.0 {
            return Err(anyhow!("invalid normalization or RoPE parameters"));
        }
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context).context("loading CUDA operator modules")?;
        let mut file = File::open(path)?;
        let mut weights = BTreeMap::new();
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
            data: DeviceBuffer::zeroed(&stream, 512 * 64)?,
            key: None,
        });
        Ok(Self {
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
                .filter(|n| [64, 128, 256].contains(n))
                .ok_or_else(|| anyhow!("LIA_MM_THREADS must be 64, 128 or 256"))?,
            attention_work: RefCell::new(None),
            prefill_input: RefCell::new(None),
            prefill_work: RefCell::new(None),
            quant_work: RefCell::new(BTreeMap::new()),
            spec_states: Vec::new(),
            capture: false,
            max_draft: std::env::var("LIA_MTP_DRAFT_MAX")
                .unwrap_or_else(|_| "3".into())
                .parse::<usize>()
                .ok()
                .filter(|n| (1..=7).contains(n))
                .ok_or_else(|| anyhow!("LIA_MTP_DRAFT_MAX must be between 1 and 7"))?,
            verify_hidden: None,
            rotary,
        })
    }
}
