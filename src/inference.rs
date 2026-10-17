//! Single-sequence Qwen 3.5 forward execution using independently written kernels.
use crate::{
    gguf::{Model, Value},
    gpu::kernels,
};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, simt::LaunchConfig};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    ops::{Deref, DerefMut},
    rc::Rc,
    sync::Arc,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type Pool = Rc<RefCell<BTreeMap<usize, Vec<DeviceBuffer<f32>>>>>;
struct Buffer {
    data: Option<DeviceBuffer<f32>>,
    pool: Pool,
}
impl Deref for Buffer {
    type Target = DeviceBuffer<f32>;
    fn deref(&self) -> &Self::Target {
        self.data.as_ref().expect("live pooled buffer")
    }
}
impl DerefMut for Buffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data.as_mut().expect("live pooled buffer")
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some(data) = self.data.take() {
            self.pool
                .borrow_mut()
                .entry(data.len())
                .or_default()
                .push(data);
        }
    }
}
struct RotaryWork {
    data: DeviceBuffer<f32>,
    key: Option<(usize, usize)>,
}
struct QuantWork {
    packed: DeviceBuffer<u32>,
    scales: DeviceBuffer<f32>,
}
struct Weight {
    data: DeviceBuffer<u8>,
    floats: Option<DeviceBuffer<f32>>,
    shape: Vec<usize>,
    kind: u32,
    block: usize,
    bytes: usize,
}
enum State {
    Recurrent {
        conv: DeviceBuffer<f32>,
        matrix: DeviceBuffer<f32>,
    },
    Attention {
        k: DeviceBuffer<u16>,
        v: DeviceBuffer<u16>,
    },
}
pub struct Snapshot {
    pub position: usize,
    recurrent: Vec<Option<(DeviceBuffer<f32>, DeviceBuffer<f32>)>>,
    hidden: Option<DeviceBuffer<f32>>,
}
pub struct Engine {
    _context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
    weights: BTreeMap<String, Weight>,
    states: Vec<State>,
    position: usize,
    capacity: usize,
    eps: f32,
    rope_base: f32,
    pool: Pool,
    last_hidden: Option<Buffer>,
    mtp_prepared: Option<(usize, u32)>,
    profile: bool,
    integer: bool,
    split_k: usize,
    row_group: usize,
    quant_work: RefCell<BTreeMap<usize, QuantWork>>,
    spec_states: Vec<Option<(DeviceBuffer<f32>, DeviceBuffer<f32>)>>,
    capture: bool,
    max_draft: usize,
    verify_hidden: Option<Buffer>,
    rotary: RefCell<RotaryWork>,
}
fn config_u64(model: &Model, key: &str) -> Result<u64> {
    match model.metadata.get(key) {
        Some(Value::Unsigned(x)) => Ok(*x),
        _ => Err(format!("missing unsigned model config {key}").into()),
    }
}
fn config_f32(model: &Model, key: &str) -> Result<f32> {
    match model.metadata.get(key) {
        Some(Value::Float(x)) => Ok(*x as f32),
        _ => Err(format!("missing float model config {key}").into()),
    }
}
fn validate_weights(model: &Model) -> Result<()> {
    let check = |name: &str, shape: &[u64], f32_only: bool| -> Result<()> {
        let tensor = model
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        if tensor.shape != shape || (f32_only && tensor.kind != 0) {
            return Err(format!("unsupported tensor layout {name}").into());
        }
        if ![0, 10, 12, 14, 16, 17, 18, 19, 21, 22, 23, 29, 30].contains(&tensor.kind) {
            return Err(format!("unsupported tensor type for {name}").into());
        }
        Ok(())
    };
    check("token_embd.weight", &[5120, 248320], false)?;
    check("output.weight", &[5120, 248320], false)?;
    check("output_norm.weight", &[5120], true)?;
    for layer in 0..65 {
        let p = format!("blk.{layer}");
        for name in ["attn_norm.weight", "post_attention_norm.weight"] {
            check(&format!("{p}.{name}"), &[5120], true)?;
        }
        for name in ["ffn_up.weight", "ffn_gate.weight"] {
            check(&format!("{p}.{name}"), &[5120, 17408], false)?;
        }
        check(&format!("{p}.ffn_down.weight"), &[17408, 5120], false)?;
        if layer == 64 || (layer + 1) % 4 == 0 {
            check(&format!("{p}.attn_q.weight"), &[5120, 12288], false)?;
            for name in ["attn_k.weight", "attn_v.weight"] {
                check(&format!("{p}.{name}"), &[5120, 1024], false)?;
            }
            check(&format!("{p}.attn_output.weight"), &[6144, 5120], false)?;
            for name in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                check(&format!("{p}.{name}"), &[256], true)?;
            }
        } else {
            check(&format!("{p}.attn_qkv.weight"), &[5120, 10240], false)?;
            check(&format!("{p}.attn_gate.weight"), &[5120, 6144], false)?;
            check(&format!("{p}.ssm_out.weight"), &[6144, 5120], false)?;
            for name in ["ssm_alpha.weight", "ssm_beta.weight"] {
                check(&format!("{p}.{name}"), &[5120, 48], false)?;
            }
            for name in ["ssm_a", "ssm_dt.bias"] {
                check(&format!("{p}.{name}"), &[48], true)?;
            }
            check(&format!("{p}.ssm_conv1d.weight"), &[4, 10240], true)?;
            check(&format!("{p}.ssm_norm.weight"), &[128], true)?;
        }
    }
    check("blk.64.nextn.eh_proj.weight", &[10240, 5120], false)?;
    for name in ["enorm.weight", "hnorm.weight", "shared_head_norm.weight"] {
        check(&format!("blk.64.nextn.{name}"), &[5120], true)?;
    }
    Ok(())
}
impl Engine {
    pub fn load(path: &str, capacity: usize) -> Result<Self> {
        if capacity == 0 || capacity > 16384 {
            return Err("context capacity must be 1..=16384".into());
        }
        let model = Model::open(path)?;
        if !matches!(model.metadata.get("general.architecture"), Some(Value::String(s)) if s == "qwen35")
        {
            return Err("only qwen35 is implemented".into());
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
            if config_u64(&model, &format!("qwen35.{key}"))? != expected {
                return Err(format!("unsupported qwen35 dimension {key}").into());
            }
        }
        validate_weights(&model)?;
        let eps = config_f32(&model, "qwen35.attention.layer_norm_rms_epsilon")?;
        let rope_base = config_f32(&model, "qwen35.rope.freq_base")?;
        if !eps.is_finite() || eps <= 0.0 || !rope_base.is_finite() || rope_base <= 0.0 {
            return Err("invalid normalization or RoPE parameters".into());
        }
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        let mut file = File::open(path)?;
        let mut weights = BTreeMap::new();
        for tensor in &model.tensors {
            let mut host = vec![0; tensor.byte_len as usize];
            file.seek(SeekFrom::Start(model.data_offset + tensor.offset))?;
            file.read_exact(&mut host)?;
            let (block, bytes) =
                crate::gguf::layout(tensor.kind).ok_or("unsupported storage layout")?;
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
            data: DeviceBuffer::zeroed(&stream, 512)?,
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
                .ok_or("LIA_SPLIT_K must be 1 or 4")?,
            row_group: std::env::var("LIA_MM_ROWS")
                .unwrap_or_else(|_| "2".into())
                .parse::<usize>()
                .ok()
                .filter(|n| *n == 1 || *n == 2)
                .ok_or("LIA_MM_ROWS must be 1 or 2")?,
            quant_work: RefCell::new(BTreeMap::new()),
            spec_states: Vec::new(),
            capture: false,
            max_draft: std::env::var("LIA_MTP_DRAFT_MAX")
                .unwrap_or_else(|_| "3".into())
                .parse::<usize>()
                .ok()
                .filter(|n| (1..=7).contains(n))
                .ok_or("LIA_MTP_DRAFT_MAX must be between 1 and 7")?,
            verify_hidden: None,
            rotary,
        })
    }
    fn weight(&self, name: &str) -> Result<&Weight> {
        self.weights
            .get(name)
            .ok_or_else(|| format!("missing tensor {name}").into())
    }
    fn floats(&self, name: &str) -> Result<&DeviceBuffer<f32>> {
        self.weight(name)?
            .floats
            .as_ref()
            .ok_or_else(|| format!("expected F32 tensor {name}").into())
    }
    fn zeros(&self, n: usize) -> Result<Buffer> {
        let recycled = self
            .pool
            .borrow_mut()
            .get_mut(&n)
            .and_then(|buffers| buffers.pop());
        let data = match recycled {
            Some(data) => data,
            None => DeviceBuffer::zeroed(&self.stream, n)?,
        };
        // Scratch kernels fully overwrite their outputs. Reuse is ordered on the
        // same stream, including when an earlier input allocation is recycled.
        Ok(Buffer {
            data: Some(data),
            pool: self.pool.clone(),
        })
    }
    fn mm(&self, name: &str, input: &DeviceBuffer<f32>) -> Result<Buffer> {
        let w = self.weight(name)?;
        if w.shape.len() != 2 || !input.len().is_multiple_of(w.shape[0]) {
            return Err(format!("matrix shape mismatch: {name}").into());
        }
        let start = if self.profile {
            self.stream.synchronize()?;
            Some(std::time::Instant::now())
        } else {
            None
        };
        let cols = w.shape[0];
        let rows = w.shape[1];
        let batch = input.len() / cols;
        let output = self.zeros(rows * batch)?;
        // SAFETY: validated matrix layout, matching input and allocated output;
        // every warp has one disjoint output row.
        if w.kind == 30 && rows <= 96 {
            let parts = cols.div_ceil(256);
            let partial = self.zeros(parts * rows * batch)?;
            // SAFETY: the checked BF16 matrix has exactly two bytes per value;
            // each partition and output row has a disjoint warp producer.
            unsafe {
                self.module.bf16_partial(
                    &self.stream,
                    LaunchConfig::for_num_elems((parts * rows * batch * 32) as u32),
                    cols,
                    rows,
                    batch,
                    parts,
                    &w.data,
                    input,
                    partial.cu_deviceptr() as *mut f32,
                )?;
                self.module.bf16_merge(
                    &self.stream,
                    LaunchConfig::for_num_elems((rows * batch * 32) as u32),
                    parts,
                    &partial,
                    output.cu_deviceptr() as *mut f32,
                )?;
            }
        } else if self.integer && w.kind != 0 && w.kind != 30 {
            let mut work = self.quant_work.borrow_mut();
            if let std::collections::btree_map::Entry::Vacant(entry) = work.entry(input.len()) {
                entry.insert(QuantWork {
                    packed: DeviceBuffer::zeroed(&self.stream, input.len() / 4)?,
                    scales: DeviceBuffer::zeroed(&self.stream, input.len() / 32)?,
                });
            }
            let work = work
                .get(&input.len())
                .ok_or("missing quantization workspace")?;
            // SAFETY: complete 32-element activation groups, correctly sized
            // persistent workspaces, and a disjoint output row per warp. Every
            // use of this workspace is ordered on the same stream.
            unsafe {
                self.module.quantize_activation(
                    &self.stream,
                    LaunchConfig::for_num_elems(input.len() as u32),
                    input,
                    work.packed.cu_deviceptr() as *mut u32,
                    work.scales.cu_deviceptr() as *mut f32,
                )?;
                crate::gpu::launch_int_matmul(
                    &self.module,
                    &self.stream,
                    w.kind,
                    cols,
                    rows,
                    batch,
                    self.split_k,
                    self.row_group,
                    &w.data,
                    &work.packed,
                    &work.scales,
                    output.cu_deviceptr() as *mut f32,
                )?;
            }
        } else {
            unsafe {
                crate::gpu::launch_matmul(
                    &self.module,
                    &self.stream,
                    w.kind,
                    cols,
                    rows,
                    batch,
                    &w.data,
                    input,
                    output.cu_deviceptr() as *mut f32,
                )?;
            }
        }
        if let Some(start) = start {
            self.stream.synchronize()?;
            eprintln!(
                "matrix {name} type={} batch={batch} rows={rows} cols={cols} seconds={}",
                w.kind,
                start.elapsed().as_secs_f64()
            );
        }
        Ok(output)
    }
    fn norm(&self, name: &str, input: &DeviceBuffer<f32>, dim: usize) -> Result<Buffer> {
        let weight = self.floats(name)?;
        if !input.len().is_multiple_of(dim) || weight.len() != dim {
            return Err(format!("norm shape mismatch: {name}").into());
        }
        let groups = input.len() / dim;
        let output = self.zeros(input.len())?;
        // SAFETY: complete disjoint groups, with matching normalization weights.
        unsafe {
            self.module.rms_norm(
                &self.stream,
                LaunchConfig::for_num_elems((groups * 32) as u32),
                dim,
                groups,
                self.eps,
                input,
                weight,
                output.cu_deviceptr() as *mut f32,
            )?;
        }
        Ok(output)
    }
    fn add(&self, a: &DeviceBuffer<f32>, b: &DeviceBuffer<f32>) -> Result<Buffer> {
        if a.len() != b.len() {
            return Err("residual length mismatch".into());
        }
        let mut out = self.zeros(a.len())?;
        // SAFETY: equally sized buffers and guarded element-wise output.
        unsafe {
            self.module.residual(
                &self.stream,
                LaunchConfig::for_num_elems(a.len() as u32),
                a,
                b,
                &mut out,
            )?;
        }
        Ok(out)
    }
    fn silu_mul(&self, x: &DeviceBuffer<f32>, gate: &DeviceBuffer<f32>) -> Result<Buffer> {
        if x.len() != gate.len() {
            return Err("gate length mismatch".into());
        }
        let mut out = self.zeros(x.len())?;
        // SAFETY: equally sized buffers and guarded element-wise output.
        unsafe {
            self.module.silu_mul(
                &self.stream,
                LaunchConfig::for_num_elems(x.len() as u32),
                x,
                gate,
                &mut out,
            )?;
        }
        Ok(out)
    }
    fn linear_attention(&self, layer: usize, x: &DeviceBuffer<f32>) -> Result<Buffer> {
        let batch = x.len() / 5120;
        let prefix = format!("blk.{layer}");
        let qkv = self.mm(&format!("{prefix}.attn_qkv.weight"), x)?;
        let z = self.mm(&format!("{prefix}.attn_gate.weight"), x)?;
        let alpha = self.mm(&format!("{prefix}.ssm_alpha.weight"), x)?;
        let beta = self.mm(&format!("{prefix}.ssm_beta.weight"), x)?;
        let State::Recurrent {
            conv: history,
            matrix,
        } = &self.states[layer]
        else {
            return Err("invalid recurrent layer state".into());
        };
        let captured = if self.capture {
            self.spec_states[layer].as_ref()
        } else {
            None
        };
        let conv = self.zeros(batch * 10240)?;
        let qk = self.zeros(batch * 4096)?;
        let core = self.zeros(batch * 6144)?;
        let decay = self.zeros(batch * 48)?;
        let bnorm = self.zeros(batch * 48)?;
        // SAFETY: model dimensions were checked at load; state rows and history
        // channels have unique writers and launches are ordered on one stream.
        unsafe {
            self.module.conv_silu(
                &self.stream,
                LaunchConfig::for_num_elems(10240),
                10240,
                &qkv,
                self.floats(&format!("{prefix}.ssm_conv1d.weight"))?,
                history.cu_deviceptr() as *mut f32,
                captured.map_or(core::ptr::null_mut(), |(c, _)| c.cu_deviceptr() as *mut f32),
                conv.cu_deviceptr() as *mut f32,
            )?;
            self.module.l2_norm(
                &self.stream,
                LaunchConfig::for_num_elems((batch * 32 * 32) as u32),
                128,
                batch * 32,
                32,
                10240,
                self.eps,
                &conv,
                qk.cu_deviceptr() as *mut f32,
            )?;
            self.module.gdn_gates(
                &self.stream,
                LaunchConfig::for_num_elems((batch * 48) as u32),
                48,
                &alpha,
                &beta,
                self.floats(&format!("{prefix}.ssm_dt.bias"))?,
                self.floats(&format!("{prefix}.ssm_a"))?,
                decay.cu_deviceptr() as *mut f32,
                bnorm.cu_deviceptr() as *mut f32,
            )?;
            self.module.delta_net_128(
                &self.stream,
                LaunchConfig::for_num_elems(6144 * 32),
                16,
                48,
                &qk,
                &conv,
                &decay,
                &bnorm,
                matrix.cu_deviceptr() as *mut f32,
                captured.map_or(core::ptr::null_mut(), |(_, m)| m.cu_deviceptr() as *mut f32),
                core.cu_deviceptr() as *mut f32,
            )?;
        }
        let normalized = self.norm(&format!("{prefix}.ssm_norm.weight"), &core, 128)?;
        let gated = self.silu_mul(&normalized, &z)?;
        self.mm(&format!("{prefix}.ssm_out.weight"), &gated)
    }
    fn full_attention(&self, layer: usize, x: &DeviceBuffer<f32>) -> Result<Buffer> {
        self.full_attention_at(layer, x, self.position, false)?
            .ok_or_else(|| "missing attention output".into())
    }
    fn full_attention_at(
        &self,
        layer: usize,
        x: &DeviceBuffer<f32>,
        position: usize,
        cache_only: bool,
    ) -> Result<Option<Buffer>> {
        let batch = x.len() / 5120;
        let prefix = format!("blk.{layer}");
        let qgate = if cache_only {
            None
        } else {
            Some(self.mm(&format!("{prefix}.attn_q.weight"), x)?)
        };
        let kraw = self.mm(&format!("{prefix}.attn_k.weight"), x)?;
        let v = self.mm(&format!("{prefix}.attn_v.weight"), x)?;
        let q = self.zeros(batch * 6144)?;
        let k = self.zeros(batch * 1024)?;
        let parts = (position + batch).div_ceil(128);
        let cap_parts = self.capacity.div_ceil(128);
        let partial = self.zeros(batch * 24 * cap_parts * 258)?;
        let attended = self.zeros(batch * 6144)?;
        let State::Attention { k: kc, v: vc } = &self.states[layer] else {
            return Err("invalid attention layer state".into());
        };
        let tokens = position + batch;
        let mut rotary = self.rotary.borrow_mut();
        if rotary.key != Some((position, batch)) {
            // SAFETY: the persistent table holds eight complete 64-dimensional
            // rotations, while the validated token batch never exceeds eight.
            unsafe {
                self.module.rope_table(
                    &self.stream,
                    LaunchConfig::for_num_elems((batch * 32) as u32),
                    64,
                    batch,
                    position,
                    self.rope_base,
                    rotary.data.cu_deviceptr() as *mut f32,
                )?;
            }
            rotary.key = Some((position, batch));
        }

        // SAFETY: dimension and extent checks at load, bounded position, matching
        // Q/KV head grouping, and disjoint warp/element output ownership.
        unsafe {
            if let Some(qgate) = &qgate {
                self.module.norm_rope_cached(
                    &self.stream,
                    LaunchConfig::for_num_elems((batch * 24 * 32) as u32),
                    256,
                    24,
                    512,
                    64,
                    self.eps,
                    qgate,
                    self.floats(&format!("{prefix}.attn_q_norm.weight"))?,
                    &rotary.data,
                    q.cu_deviceptr() as *mut f32,
                )?;
            }
            self.module.norm_rope_cached(
                &self.stream,
                LaunchConfig::for_num_elems((batch * 4 * 32) as u32),
                256,
                4,
                256,
                64,
                self.eps,
                &kraw,
                self.floats(&format!("{prefix}.attn_k_norm.weight"))?,
                &rotary.data,
                k.cu_deviceptr() as *mut f32,
            )?;
            self.module.cache_kv_half(
                &self.stream,
                LaunchConfig::for_num_elems((batch * 1024) as u32),
                1024,
                position,
                &k,
                &v,
                kc.cu_deviceptr() as *mut u16,
                vc.cu_deviceptr() as *mut u16,
            )?;
            if cache_only {
                return Ok(None);
            }
            self.module.flash_partial_half::<256, 128>(
                &self.stream,
                LaunchConfig::for_num_elems((batch * 24 * parts * 32) as u32),
                24,
                4,
                tokens,
                parts,
                cap_parts,
                &q,
                kc,
                vc,
                partial.cu_deviceptr() as *mut f32,
            )?;
            self.module.flash_merge::<256>(
                &self.stream,
                LaunchConfig::for_num_elems((batch * 24 * 32) as u32),
                batch * 24,
                parts,
                cap_parts,
                &partial,
                qgate.as_ref().ok_or("missing attention query")?,
                attended.cu_deviceptr() as *mut f32,
            )?;
        }
        Ok(Some(
            self.mm(&format!("{prefix}.attn_output.weight"), &attended)?,
        ))
    }
    fn embed(&self, token: u32) -> Result<Buffer> {
        let emb = self.weight("token_embd.weight")?;
        if emb.shape != [5120, 248320] || token as usize >= emb.shape[1] {
            return Err("invalid vocabulary or token".into());
        }
        let mut hidden = self.zeros(5120)?;
        // SAFETY: validated vocabulary index and embedding row extent.
        unsafe {
            self.module.embedding(
                &self.stream,
                LaunchConfig::for_num_elems(5120),
                emb.kind,
                emb.bytes,
                emb.block,
                5120,
                token as usize,
                &emb.data,
                &mut hidden,
            )?;
        }
        Ok(hidden)
    }
    fn mtp_given(
        &self,
        token: u32,
        h: &DeviceBuffer<f32>,
        position: usize,
        logits: bool,
    ) -> Result<(Buffer, Option<Buffer>)> {
        let embedding = self.embed(token)?;
        let en = self.norm("blk.64.nextn.enorm.weight", &embedding, 5120)?;
        let hn = self.norm("blk.64.nextn.hnorm.weight", h, 5120)?;
        let mut concatenated = self.zeros(10240)?;
        // SAFETY: output exactly covers both nonoverlapping input vectors.
        unsafe {
            self.module.concat(
                &self.stream,
                LaunchConfig::for_num_elems(10240),
                &en,
                &hn,
                &mut concatenated,
            )?;
        }
        let projected = self.mm("blk.64.nextn.eh_proj.weight", &concatenated)?;
        let norm = self.norm("blk.64.attn_norm.weight", &projected, 5120)?;
        let Some(attention) = self.full_attention_at(64, &norm, position, !logits)? else {
            // Priming stores K/V for a known token. Its output is not a draft
            // hidden state and is intentionally discarded by the caller.
            return Ok((projected, None));
        };
        let residual = self.add(&projected, &attention)?;
        let normalized = self.norm("blk.64.post_attention_norm.weight", &residual, 5120)?;
        let up = self.mm("blk.64.ffn_up.weight", &normalized)?;
        let gate = self.mm("blk.64.ffn_gate.weight", &normalized)?;
        let activated = self.silu_mul(&up, &gate)?;
        let down = self.mm("blk.64.ffn_down.weight", &activated)?;
        let hidden = self.add(&residual, &down)?;
        let normalized = self.norm("blk.64.nextn.shared_head_norm.weight", &hidden, 5120)?;
        let result = if logits {
            Some(self.mm("output.weight", &normalized)?)
        } else {
            None
        };
        Ok((normalized, result))
    }
    fn mtp(&self, token: u32, logits: bool) -> Result<Option<Vec<f32>>> {
        let zero = if self.last_hidden.is_none() {
            Some(DeviceBuffer::<f32>::zeroed(&self.stream, 5120)?)
        } else {
            None
        };
        let h: &DeviceBuffer<f32> = match &self.last_hidden {
            Some(h) => h,
            None => zero.as_ref().ok_or("missing initial hidden state")?,
        };
        self.mtp_given(token, h, self.position, logits)?
            .1
            .map(|x| Ok(x.to_host_vec(&self.stream)?))
            .transpose()
    }
    /// Run the model's built-in MTP block on a known next token and the most
    /// recent target hidden state. The returned distribution predicts one further token.
    pub fn draft(&mut self, next_token: u32) -> Result<Vec<f32>> {
        if self.position >= self.capacity {
            return Err("MTP context capacity exceeded".into());
        }
        let result = self.mtp(next_token, true)?.ok_or("missing MTP output")?;
        self.mtp_prepared = Some((self.position, next_token));
        Ok(result)
    }
    fn greedy(&self, logits: &DeviceBuffer<f32>) -> Result<Vec<u32>> {
        const VOCAB: usize = 248320;
        if logits.is_empty() || !logits.len().is_multiple_of(VOCAB) {
            return Err("invalid greedy logit shape".into());
        }
        let parts = VOCAB.div_ceil(1024);
        let count = logits.len() / VOCAB * parts;
        let values = self.zeros(count)?;
        let ids = DeviceBuffer::<u32>::zeroed(&self.stream, count)?;
        // SAFETY: one complete warp per disjoint partition, matching partial
        // value and index arrays. Each input row covers the full vocabulary.
        unsafe {
            self.module.argmax_partial(
                &self.stream,
                LaunchConfig::for_num_elems((count * 32) as u32),
                VOCAB,
                parts,
                logits,
                values.cu_deviceptr() as *mut f32,
                ids.cu_deviceptr() as *mut u32,
            )?;
        }
        let values = values.to_host_vec(&self.stream)?;
        let ids = ids.to_host_vec(&self.stream)?;
        values
            .chunks(parts)
            .zip(ids.chunks(parts))
            .map(|(v, i)| {
                if v.iter().any(|x| !x.is_finite()) {
                    return Err("nonfinite greedy logits".into());
                }
                let best = v
                    .iter()
                    .zip(i)
                    .max_by(|a, b| a.0.total_cmp(b.0).then(a.1.cmp(b.1)))
                    .ok_or("empty greedy partition")?;
                Ok(*best.1)
            })
            .collect()
    }
    fn draft_chain(&mut self, first: u32, maximum: usize, eos: u32) -> Result<Vec<u32>> {
        let mut proposals = Vec::new();
        let mut token = first;
        let mut chain: Option<Buffer> = None;
        for step in 0..maximum {
            if self.position + step >= self.capacity {
                break;
            }
            let h: &DeviceBuffer<f32> = match &chain {
                Some(h) => h,
                None => self
                    .last_hidden
                    .as_ref()
                    .ok_or("missing target hidden state")?,
            };
            let (hidden, logits) = self.mtp_given(token, h, self.position + step, true)?;
            let logits = logits.ok_or("missing draft logits")?;
            token = self.greedy(&logits)?[0];
            chain = Some(hidden);
            proposals.push(token);
            if token == eos {
                break;
            }
        }
        self.mtp_prepared = None;
        Ok(proposals)
    }
    /// Verify the configured native MTP proposals in one target batch. Rejected
    /// suffixes restore the captured recurrent state at the accepted position.
    /// The returned logits belong to the last accepted target state.
    pub fn speculate(
        &mut self,
        first: u32,
        maximum: usize,
        eos: u32,
    ) -> Result<(Vec<u32>, u32, usize, usize)> {
        if maximum == 0 || first == eos {
            return Err("invalid speculative iteration".into());
        }
        let maximum = maximum.min(self.capacity - self.position);
        if maximum == 0 {
            return Err("context capacity exceeded".into());
        }
        if self.spec_states.is_empty() {
            for state in &self.states {
                self.spec_states.push(match state {
                    State::Recurrent { conv, matrix } => Some((
                        DeviceBuffer::zeroed(&self.stream, (self.max_draft + 1) * conv.len())?,
                        DeviceBuffer::zeroed(&self.stream, (self.max_draft + 1) * matrix.len())?,
                    )),
                    State::Attention { .. } => None,
                });
            }
        }
        let start_position = self.position;
        let drafts = self.draft_chain(first, maximum.saturating_sub(1).min(self.max_draft), eos)?;
        let mut inputs = vec![first];
        inputs.extend_from_slice(&drafts);
        self.capture = true;
        let verification = self.forward_many_device(&inputs, true);
        self.capture = false;
        let verified = verification?.ok_or("missing verification logits")?;
        let verified = self.greedy(&verified)?;
        let mut accepted = 0;
        for (index, proposal) in drafts.iter().enumerate() {
            if verified[index] != *proposal {
                break;
            }
            accepted += 1;
            if *proposal == eos {
                break;
            }
        }
        let count = 1 + accepted;
        let logits = verified[accepted];
        if count < inputs.len() {
            for (state, captured) in self.states.iter().zip(&self.spec_states) {
                if let (State::Recurrent { conv, matrix }, Some((c, m))) = (state, captured) {
                    // SAFETY: each captured allocation contains a complete state per verified position;
                    // accepted is an index into the verified batch of at most eight.
                    unsafe {
                        self.module.restore_position(
                            &self.stream,
                            LaunchConfig::for_num_elems(conv.len() as u32),
                            accepted * conv.len(),
                            conv.len(),
                            c,
                            conv.cu_deviceptr() as *mut f32,
                        )?;
                        self.module.restore_position(
                            &self.stream,
                            LaunchConfig::for_num_elems(matrix.len() as u32),
                            accepted * matrix.len(),
                            matrix.len(),
                            m,
                            matrix.cu_deviceptr() as *mut f32,
                        )?;
                    }
                }
            }
            let hidden = self
                .verify_hidden
                .as_ref()
                .ok_or("missing verified hidden states")?;
            let last = self.zeros(5120)?;
            // SAFETY: the selected complete hidden row belongs to an accepted token.
            unsafe {
                self.module.restore_position(
                    &self.stream,
                    LaunchConfig::for_num_elems(5120),
                    accepted * 5120,
                    5120,
                    hidden,
                    last.cu_deviceptr() as *mut f32,
                )?;
            }
            self.last_hidden = Some(last);
            self.position = start_position + count;
            self.mtp_prepared = None;
        }
        inputs.truncate(count);
        self.verify_hidden = None;
        Ok((inputs, logits, drafts.len(), accepted))
    }
    fn embed_many(&self, tokens: &[u32]) -> Result<Buffer> {
        if tokens.is_empty() || tokens.len() > 8 || tokens.iter().any(|t| *t >= 248320) {
            return Err("invalid token batch".into());
        }
        let w = self.weight("token_embd.weight")?;
        let ids = DeviceBuffer::from_host(&self.stream, tokens)?;
        let mut out = self.zeros(tokens.len() * 5120)?;
        // SAFETY: validated token indices, row-aligned embedding table and
        // exactly one guarded output per token embedding element.
        unsafe {
            self.module.embedding_many(
                &self.stream,
                LaunchConfig::for_num_elems((tokens.len() * 5120) as u32),
                w.kind,
                w.bytes,
                w.block,
                5120,
                &ids,
                &w.data,
                &mut out,
            )?;
        }
        Ok(out)
    }
    fn mtp_many(&self, tokens: &[u32], target: &DeviceBuffer<f32>) -> Result<()> {
        let batch = tokens.len();
        let embedding = self.embed_many(tokens)?;
        let zero = if self.last_hidden.is_none() {
            Some(DeviceBuffer::<f32>::zeroed(&self.stream, 5120)?)
        } else {
            None
        };
        let prev: &DeviceBuffer<f32> = match &self.last_hidden {
            Some(h) => h,
            None => zero.as_ref().ok_or("missing initial MTP hidden")?,
        };
        let mut shifted = self.zeros(batch * 5120)?;
        // SAFETY: the first shifted row comes from the previous target state,
        // later rows from this batch's preceding states; all extents match.
        unsafe {
            self.module.shift_hidden(
                &self.stream,
                LaunchConfig::for_num_elems((batch * 5120) as u32),
                5120,
                prev,
                target,
                &mut shifted,
            )?;
        }
        let en = self.norm("blk.64.nextn.enorm.weight", &embedding, 5120)?;
        let hn = self.norm("blk.64.nextn.hnorm.weight", &shifted, 5120)?;
        let mut concatenated = self.zeros(batch * 10240)?;
        // SAFETY: corresponding token rows are concatenated into disjoint rows.
        unsafe {
            self.module.concat_rows(
                &self.stream,
                LaunchConfig::for_num_elems((batch * 10240) as u32),
                5120,
                &en,
                &hn,
                &mut concatenated,
            )?;
        }
        let projected = self.mm("blk.64.nextn.eh_proj.weight", &concatenated)?;
        let normalized = self.norm("blk.64.attn_norm.weight", &projected, 5120)?;
        // Known-prefix priming only needs this block's K/V cache. The MTP
        // attention output and FFN have no persistent state and are unused.
        self.full_attention_at(64, &normalized, self.position, true)?;
        Ok(())
    }
    pub fn forward_many(&mut self, tokens: &[u32], want_logits: bool) -> Result<Vec<f32>> {
        match self.forward_many_device(tokens, want_logits)? {
            Some(logits) => Ok(logits.to_host_vec(&self.stream)?),
            None => Ok(Vec::new()),
        }
    }
    fn forward_many_device(&mut self, tokens: &[u32], want_logits: bool) -> Result<Option<Buffer>> {
        if tokens.is_empty() || tokens.len() > 8 || self.position + tokens.len() > self.capacity {
            return Err("invalid forward batch or exhausted context".into());
        }
        let batch = tokens.len();
        let mut hidden = self.embed_many(tokens)?;
        for layer in 0..64 {
            let normalized = self.norm(&format!("blk.{layer}.attn_norm.weight"), &hidden, 5120)?;
            let attention = if (layer + 1) % 4 == 0 {
                self.full_attention(layer, &normalized)?
            } else {
                self.linear_attention(layer, &normalized)?
            };
            let residual = self.add(&hidden, &attention)?;
            let normalized = self.norm(
                &format!("blk.{layer}.post_attention_norm.weight"),
                &residual,
                5120,
            )?;
            let up = self.mm(&format!("blk.{layer}.ffn_up.weight"), &normalized)?;
            let gate = self.mm(&format!("blk.{layer}.ffn_gate.weight"), &normalized)?;
            let activated = self.silu_mul(&up, &gate)?;
            let down = self.mm(&format!("blk.{layer}.ffn_down.weight"), &activated)?;
            hidden = self.add(&residual, &down)?;
        }
        let normalized = self.norm("output_norm.weight", &hidden, 5120)?;
        let logits = if want_logits {
            Some(self.mm("output.weight", &normalized)?)
        } else {
            None
        };
        self.mtp_many(tokens, &normalized)?;
        let mut last = self.zeros(5120)?;
        // SAFETY: a complete final token row is copied to a fresh hidden buffer.
        unsafe {
            self.module.last_hidden(
                &self.stream,
                LaunchConfig::for_num_elems(5120),
                5120,
                &normalized,
                &mut last,
            )?;
        }
        self.last_hidden = Some(last);
        if self.capture {
            self.verify_hidden = Some(normalized);
        }
        self.position += batch;
        self.mtp_prepared = None;
        Ok(logits)
    }
    pub fn forward(&mut self, token: u32) -> Result<Vec<f32>> {
        self.step(token, true)
    }
    fn step(&mut self, token: u32, want_logits: bool) -> Result<Vec<f32>> {
        if self.position >= self.capacity {
            return Err("context capacity exceeded".into());
        }
        if self.mtp_prepared != Some((self.position, token)) {
            self.mtp(token, false)?;
        }
        self.mtp_prepared = Some((self.position, token));
        let mut hidden = self.embed(token)?;
        for layer in 0..64 {
            let normalized = self.norm(&format!("blk.{layer}.attn_norm.weight"), &hidden, 5120)?;
            let attention = if (layer + 1) % 4 == 0 {
                self.full_attention(layer, &normalized)?
            } else {
                self.linear_attention(layer, &normalized)?
            };
            let residual = self.add(&hidden, &attention)?;
            let normalized = self.norm(
                &format!("blk.{layer}.post_attention_norm.weight"),
                &residual,
                5120,
            )?;
            let up = self.mm(&format!("blk.{layer}.ffn_up.weight"), &normalized)?;
            let gate = self.mm(&format!("blk.{layer}.ffn_gate.weight"), &normalized)?;
            let activated = self.silu_mul(&up, &gate)?;
            let down = self.mm(&format!("blk.{layer}.ffn_down.weight"), &activated)?;
            hidden = self.add(&residual, &down)?;
        }
        let normalized = self.norm("output_norm.weight", &hidden, 5120)?;
        let logits = if want_logits {
            self.mm("output.weight", &normalized)?
                .to_host_vec(&self.stream)?
        } else {
            Vec::new()
        };
        self.last_hidden = Some(normalized);
        self.position += 1;
        if logits.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite model logits".into());
        }
        Ok(logits)
    }
    pub fn max_draft(&self) -> usize {
        self.max_draft
    }
    pub fn matrix_layout(&self) -> (usize, usize) {
        (self.split_k, self.row_group)
    }
    pub fn integer_enabled(&self) -> bool {
        self.integer
    }
    pub fn current_token(&self) -> Result<u32> {
        let hidden = self.last_hidden.as_ref().ok_or("no target hidden state")?;
        let logits = self.mm("output.weight", hidden)?;
        Ok(self.greedy(&logits)?[0])
    }
    fn copy(&self, input: &DeviceBuffer<f32>) -> Result<DeviceBuffer<f32>> {
        let mut out = DeviceBuffer::zeroed(&self.stream, input.len())?;
        out.copy_from_device_async(input, &self.stream)?;
        Ok(out)
    }
    pub fn refresh_snapshot(&self, previous: Option<Snapshot>) -> Result<Snapshot> {
        let Some(mut saved) = previous else {
            return self.snapshot();
        };
        for (state, dest) in self.states.iter().zip(&mut saved.recurrent) {
            if let (State::Recurrent { conv, matrix }, Some((dc, dm))) = (state, dest) {
                dc.copy_from_device_async(conv, &self.stream)?;
                dm.copy_from_device_async(matrix, &self.stream)?;
            }
        }
        match (&self.last_hidden, &mut saved.hidden) {
            (Some(source), Some(dest)) => dest.copy_from_device_async(source, &self.stream)?,
            (Some(source), None) => saved.hidden = Some(self.copy(source)?),
            (None, _) => saved.hidden = None,
        }
        saved.position = self.position;
        self.stream.synchronize()?;
        Ok(saved)
    }
    pub fn snapshot(&self) -> Result<Snapshot> {
        let mut recurrent = Vec::new();
        for state in &self.states {
            recurrent.push(match state {
                State::Recurrent { conv, matrix } => Some((self.copy(conv)?, self.copy(matrix)?)),
                State::Attention { .. } => None,
            });
        }
        let hidden = match &self.last_hidden {
            Some(h) => Some(self.copy(h)?),
            None => None,
        };
        self.stream.synchronize()?;
        Ok(Snapshot {
            position: self.position,
            recurrent,
            hidden,
        })
    }
    pub fn restore(&mut self, snapshot: &Snapshot) -> Result<()> {
        for (state, saved) in self.states.iter_mut().zip(&snapshot.recurrent) {
            if let (State::Recurrent { conv, matrix }, Some((sc, sm))) = (state, saved) {
                conv.copy_from_device_async(sc, &self.stream)?;
                matrix.copy_from_device_async(sm, &self.stream)?;
            }
        }
        self.last_hidden = match &snapshot.hidden {
            Some(h) => {
                let mut out = self.zeros(h.len())?;
                out.copy_from_device_async(h, &self.stream)?;
                Some(out)
            }
            None => None,
        };
        self.position = snapshot.position;
        self.mtp_prepared = None;
        self.stream.synchronize()?;
        Ok(())
    }
    pub fn reset(&mut self) -> Result<()> {
        self.stream.synchronize()?;
        for state in &mut self.states {
            match state {
                State::Recurrent { conv, matrix } => {
                    *conv = DeviceBuffer::zeroed(&self.stream, conv.len())?;
                    *matrix = DeviceBuffer::zeroed(&self.stream, matrix.len())?;
                }
                State::Attention { .. } => {}
            }
        }
        self.position = 0;
        self.last_hidden = None;
        self.mtp_prepared = None;
        Ok(())
    }
}
