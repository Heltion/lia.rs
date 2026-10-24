use super::*;

impl Engine {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn initialize(
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        module: kernels::LoadedModule,
        mut weights: BTreeMap<String, Weight>,
        capacity: usize,
        geometry: Geometry,
        eps: f32,
        rope_base: f32,
        fp8_prefill: bool,
        cuda_graph: bool,
    ) -> Result<Self> {
        let host_vision = if vision::offload::enabled() {
            vision::offload::park_weights(&mut weights, &stream)?
        } else {
            BTreeMap::new()
        };
        let mut states = Vec::new();
        for layer in 0..geometry.layers + usize::from(geometry.mtp) {
            states.push(
                if geometry.llama
                    || geometry.qwen3
                    || layer == geometry.layers
                    || (layer + 1) % 4 == 0
                {
                    State::Attention {
                        k: DeviceBuffer::zeroed(
                            &stream,
                            capacity
                                * geometry.kv_heads
                                * if geometry.llama || geometry.qwen3 {
                                    128
                                } else {
                                    256
                                },
                        )?,
                        v: DeviceBuffer::zeroed(
                            &stream,
                            capacity
                                * geometry.kv_heads
                                * if geometry.llama || geometry.qwen3 {
                                    128
                                } else {
                                    256
                                },
                        )?,
                    }
                } else {
                    State::Recurrent {
                        conv: DeviceBuffer::zeroed(&stream, 3 * geometry.recurrent_width())?,
                        matrix: DeviceBuffer::zeroed(&stream, geometry.value_heads * 128 * 128)?,
                    }
                },
            );
        }
        stream.synchronize()?;
        let rotary = RefCell::new(RotaryWork {
            data: DeviceBuffer::zeroed(&stream, crate::gpu::MAX_PREFILL_BATCH * 64)?,
            key: None,
            positions: DeviceBuffer::zeroed(&stream, capacity * 4)?,
            plan: None,
        });
        let token_ids = DeviceBuffer::zeroed(&stream, crate::gpu::MAX_PREFILL_BATCH)?;
        let zero_hidden = DeviceBuffer::zeroed(&stream, geometry.hidden)?;
        let mut engine = Self {
            geometry,
            awq_weights: BTreeMap::new(),
            experts: BTreeMap::new(),
            routing: if geometry.moe {
                Some(RoutingWork {
                    router_count: DeviceBuffer::zeroed(&stream, 8)?,
                    register: if std::env::var("LIA_MOE_REGISTER_MMA").as_deref() == Ok("1") {
                        Some(crate::gpu::moe::RegisterWork::new(
                            &stream,
                            8,
                            geometry.hidden,
                            512,
                            std::env::var("LIA_MOE_STAGED").as_deref() == Ok("1"),
                        )?)
                    } else {
                        None
                    },
                    grouped: crate::gpu::moe::Grouped::new(&stream, crate::gpu::MAX_PREFILL_BATCH)?,
                    selected: DeviceBuffer::zeroed(&stream, crate::gpu::MAX_PREFILL_BATCH * 8)?,
                    probabilities: DeviceBuffer::zeroed(
                        &stream,
                        crate::gpu::MAX_PREFILL_BATCH * 8,
                    )?,
                })
            } else {
                None
            },
            presence: None,
            presence_strength: 0.0,
            _context: context,
            stream: stream.clone(),
            module,
            weights,
            host_weight_bytes: 0,
            states,
            position: 0,
            paging: None,
            kv_batch_copy: Cell::new(std::env::var("LIA_KV_BATCH_COPY").as_deref() != Ok("0")),
            paged_replays: RefCell::new(BTreeMap::new()),
            capacity,
            eps,
            rope_base,
            pool: Rc::new(RefCell::new(BTreeMap::new())),
            cache_pool: Rc::new(RefCell::new(BTreeMap::new())),
            last_hidden: None,
            mtp_prepared: None,
            profile: std::env::var_os("LIA_PROFILE").is_some(),
            integer: std::env::var("LIA_INT8").as_deref().unwrap_or("1") != "0",
            bf16_vector: std::env::var("LIA_BF16_VECTOR").as_deref() != Ok("0"),
            llama_flash: std::env::var("LIA_LLAMA_FLASH").as_deref() != Ok("0"),
            llama_flash_wide: std::env::var("LIA_LLAMA_FLASH_WIDE").as_deref() == Ok("1"),
            qwen3_vector_norm: std::env::var("LIA_QWEN3_VECTOR_NORM").as_deref() == Ok("1"),
            qwen3_register_p: std::env::var("LIA_QWEN3_REGISTER_P").as_deref() == Ok("1"),
            qwen3_tensor_pv: std::env::var("LIA_QWEN3_TENSOR_PV").as_deref() == Ok("1"),
            qwen3_prepare_fusion: std::env::var("LIA_QWEN3_PREPARE_FUSION").as_deref() == Ok("1"),
            llama_fusion: std::env::var("LIA_LLAMA_FUSION").as_deref() != Ok("0"),
            llama_gated: std::env::var("LIA_LLAMA_GATED").as_deref() != Ok("0"),
            llama_weight_cache: BTreeMap::new(),
            llama_full_prefill_span: std::env::var("LIA_LLAMA_FULL_PREFILL_SPAN").as_deref()
                == Ok("1"),
            llama_flash_decode: std::env::var("LIA_LLAMA_FLASH_DECODE").as_deref() != Ok("0"),
            nucleus: RefCell::new(None),
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
            attention_decode_span: std::env::var("LIA_ATTN_DECODE_SPAN")
                .unwrap_or_else(|_| "128".into())
                .parse::<usize>()
                .ok()
                .filter(|n| [128, 256].contains(n) || (geometry.qwen3 && [32, 64].contains(n)))
                .ok_or_else(|| {
                    anyhow!(
                        "LIA_ATTN_DECODE_SPAN must be 128 or 256 (Qwen3 also supports 32 and 64)"
                    )
                })?,
            attention_merge_shared: std::env::var("LIA_ATTN_MERGE_SHARED").as_deref() == Ok("1"),
            parallel_convolution: std::env::var("LIA_CONV_PARALLEL").as_deref() == Ok("1"),
            attention_mma_decode: std::env::var("LIA_ATTN_MMA_DECODE")
                .map_or(geometry.moe, |value| value == "1"),
            attention_query_tiles: if std::env::var("LIA_ATTN_QUERY_TILES").as_deref() == Ok("1") {
                std::env::var("LIA_ATTN_QUERY_ROWS")
                    .unwrap_or_else(|_| "32".into())
                    .parse::<usize>()
                    .ok()
                    .filter(|n| [32, 64].contains(n))
                    .ok_or_else(|| anyhow!("LIA_ATTN_QUERY_ROWS must be 32 or 64"))?
            } else {
                0
            },
            attention_work: RefCell::new(None),
            prefill_batch: std::env::var("LIA_PREFILL_BATCH")
                .unwrap_or_else(|_| "512".into())
                .parse::<usize>()
                .ok()
                .filter(|batch| [128, 256, 512, 1024, 2048, 4096].contains(batch))
                .ok_or_else(|| {
                    anyhow!("LIA_PREFILL_BATCH must be 128, 256, 512, 1024, 2048, or 4096")
                })?,
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
            decode_replay: RefCell::new(None),
            decode_pool: Rc::new(RefCell::new(BTreeMap::new())),
            decode_pool_active: Rc::new(Cell::new(false)),
            qwen3_decode_replay: RefCell::new(None),
            qwen3_replay_enabled: std::env::var("LIA_QWEN3_DECODE_REPLAY").as_deref() == Ok("1"),
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
                .filter(|n| (0..=7).contains(n))
                .ok_or_else(|| anyhow!("LIA_MTP_DRAFT_MAX must be between 0 and 7"))?,
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
            vision_encoder: crate::models::VisionEncoder::Qwen3Vl,
            host_vision,
            kv_offloaded: false,
            embeddings: Vec::new(),
        };
        engine.geometry.mtp &= engine.max_draft > 0;
        if !engine.geometry.mtp {
            engine.max_draft = 0;
        }
        engine.stochastic_mtp &= engine.geometry.mtp;
        if engine.geometry.mtp {
            engine.prepare_speculation_buffers()?;
        }
        engine.prepare_small_workspaces()?;
        if std::env::var("LIA_GDN_PARALLEL").as_deref() == Ok("1") {
            engine.gdn_prepared = Some(DeviceBuffer::zeroed(
                &engine.stream,
                geometry.value_heads * crate::gpu::MAX_PREFILL_BATCH.div_ceil(16) * 1072,
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
        Ok(engine)
    }
}
