//! Single-sequence Qwen 3.5 forward execution using independently written kernels.
use crate::{
    gguf::{Model, Value},
    gpu::kernels,
};
use anyhow::{Result, anyhow};
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
mod allocation;
mod attention_plan;
mod buffer;
mod graph;
use buffer::Buffer;
struct RotaryWork {
    data: DeviceBuffer<f32>,
    key: Option<(usize, usize)>,
    positions: DeviceBuffer<u32>,
    plan: Option<crate::vision::PositionPlan>,
}
struct Fp8Work {
    weight_tiles: Option<DeviceBuffer<u32>>,
    maxima: DeviceBuffer<f32>,
    scales: DeviceBuffer<f32>,
    packed: DeviceBuffer<u32>,
}
struct QuantWork {
    packed: DeviceBuffer<u32>,
    scales: DeviceBuffer<f32>,
}
struct SelectionWork {
    candidates: DeviceBuffer<u64>,
}
struct Weight {
    data: DeviceBuffer<u8>,
    floats: Option<DeviceBuffer<f32>>,
    shape: Vec<usize>,
    kind: u32,
    fp8_scale: Option<f32>,
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
    positions: Option<crate::vision::PositionPlan>,
    images: Vec<Rc<vision::ImageEmbedding>>,
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
    pool: Rc<RefCell<BTreeMap<usize, Vec<DeviceBuffer<f32>>>>>,
    last_hidden: Option<Buffer>,
    mtp_prepared: Option<(usize, u32)>,
    profile: bool,
    preallocated: bool,
    cuda_graph: bool,
    graphs: RefCell<BTreeMap<(u8, usize, bool, bool), graph::Executable>>,
    graph_stats: RefCell<[u64; 3]>,
    scratch_reserved: usize,
    token_ids: DeviceBuffer<u32>,
    zero_hidden: DeviceBuffer<f32>,
    gdn_prepared: Option<DeviceBuffer<u32>>,
    greedy_work: BTreeMap<usize, DeviceBuffer<u32>>,
    integer: bool,
    split_k: usize,
    row_group: usize,
    matrix_threads: usize,
    attention_mma_decode: bool,
    attention_query_tiles: bool,
    attention_work: RefCell<Option<DeviceBuffer<f32>>>,
    prefill_tile: usize,
    prefill_batch: usize,
    fp8_work: RefCell<Option<Fp8Work>>,
    prefill_input: RefCell<Option<DeviceBuffer<u32>>>,
    quant_work: RefCell<BTreeMap<usize, QuantWork>>,
    sampling_work: RefCell<BTreeMap<usize, SelectionWork>>,
    spec_states: Vec<Option<(DeviceBuffer<f32>, DeviceBuffer<f32>)>>,
    capture: bool,
    max_draft: usize,
    stochastic_mtp: bool,
    verify_hidden: Option<Buffer>,
    rotary: RefCell<RotaryWork>,
    vision_eps: Option<f32>,
    images: Vec<Rc<vision::ImageEmbedding>>,
}
fn config_u64(model: &Model, key: &str) -> Result<u64> {
    match model.metadata.get(key) {
        Some(Value::Unsigned(x)) => Ok(*x),
        _ => Err(anyhow!("missing unsigned model config {key}")),
    }
}
fn config_f32(model: &Model, key: &str) -> Result<f32> {
    match model.metadata.get(key) {
        Some(Value::Float(x)) => Ok(*x as f32),
        _ => Err(anyhow!("missing float model config {key}")),
    }
}
fn validate_weights(model: &Model) -> Result<()> {
    let check = |name: &str, shape: &[u64], f32_only: bool| -> Result<()> {
        let tensor = model
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| anyhow!("missing tensor {name}"))?;
        if tensor.shape != shape || (f32_only && tensor.kind != 0) {
            return Err(anyhow!("unsupported tensor layout {name}"));
        }
        if ![0, 10, 12, 14, 16, 17, 18, 19, 21, 22, 23, 29, 30].contains(&tensor.kind) {
            return Err(anyhow!("unsupported tensor type for {name}"));
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

mod attention;
mod batch;
mod cache;
mod forward;
mod load;
mod mtp;
mod operations;
mod positions;
mod prefill;
mod projections;
mod recurrent;
mod sampling;
mod speculation;
mod vision;
