//! Qwen3 decode replay owns isolated scratch and validates cache identity and geometry.
use super::*;
use std::ffi::CStr;

pub(in crate::inference) struct Replay {
    pub(super) exec: Executable,
    _graph: Graph,
    scalars: Vec<(bool, scalar::Scalar)>,
    caches: Vec<(u64, u64)>,
    hidden: u64,
    next: usize,
    layout: (usize, usize),
}

fn layout(engine: &Engine, position: usize) -> (usize, usize) {
    if position >= engine.attention_decode_span {
        let span = engine.attention_decode_span;
        (span, (position + 1).div_ceil(span))
    } else {
        (0, 1)
    }
}

impl Replay {
    pub(super) fn new(graph: Graph, engine: &Engine) -> Result<Self> {
        let position = engine.position - 1;
        let mut count = 0;
        // SAFETY: completed kernel-only capture remains owned for this replay's lifetime.
        check(
            unsafe { sys::cuGraphGetNodes(graph.0, std::ptr::null_mut(), &mut count) },
            "counting Qwen3 graph nodes",
        )?;
        let mut nodes = vec![std::ptr::null_mut(); count];
        check(
            unsafe { sys::cuGraphGetNodes(graph.0, nodes.as_mut_ptr(), &mut count) },
            "reading Qwen3 graph nodes",
        )?;
        let (mut embeddings, mut prepares, mut attentions) = (0, 0, 0);
        let mut scalars = Vec::new();
        for node in nodes {
            let mut kind = 0;
            check(
                unsafe { sys::cuGraphNodeGetType(node, &mut kind) },
                "reading Qwen3 graph node kind",
            )?;
            anyhow::ensure!(
                kind == sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL,
                "Qwen3 decode replay requires a kernel-only graph"
            );
            let mut params: sys::CUDA_KERNEL_NODE_PARAMS_v2 = unsafe { std::mem::zeroed() };
            check(
                unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut params) },
                "reading Qwen3 captured kernel",
            )?;
            let mut name = std::ptr::null();
            check(
                unsafe { sys::cuFuncGetName(&mut name, params.func) },
                "reading Qwen3 kernel name",
            )?;
            let name = unsafe { CStr::from_ptr(name) }.to_str()?;
            let change = match name {
                "embedding" => {
                    embeddings += 1;
                    Some((true, 4, None))
                }
                "qwen3_prepare" => {
                    prepares += 1;
                    Some((false, 1, Some(position)))
                }
                "dense_attention128" => {
                    attentions += 1;
                    Some((false, 3, Some(position)))
                }
                _ => None,
            };
            if let Some((token, index, expected)) = change {
                scalars.push((token, scalar::Scalar::new(node, params, index, expected)?));
            }
        }
        anyhow::ensure!(
            embeddings == 1
                && prepares == engine.geometry.layers
                && attentions == engine.geometry.layers,
            "unexpected Qwen3 graph topology: embeddings={embeddings} prepares={prepares} attentions={attentions}"
        );
        let caches = engine
            .states
            .iter()
            .map(|state| match state {
                State::Attention { k, v } => Ok((k.cu_deviceptr(), v.cu_deviceptr())),
                _ => anyhow::bail!("Qwen3 replay requires dense attention states"),
            })
            .collect::<Result<Vec<_>>>()?;
        let hidden = engine
            .last_hidden
            .as_ref()
            .context("missing Qwen3 graph output")?
            .cu_deviceptr();
        let exec = Executable::instantiate(&graph, engine._context.clone())?;
        Ok(Self {
            exec,
            _graph: graph,
            scalars,
            caches,
            hidden,
            next: position + 1,
            layout: layout(engine, position),
        })
    }
}

pub(in crate::inference) struct Scratch(Rc<Cell<bool>>);
impl Drop for Scratch {
    fn drop(&mut self) {
        self.0.set(false);
    }
}

impl Engine {
    fn qwen3_replay_eligible(&self, want_logits: bool) -> bool {
        self.qwen3_replay_enabled
            && self.cuda_graph
            && self.preallocated
            && self.geometry.qwen3
            && !self.geometry.mtp
            && !self.capture
            && !want_logits
            && self.paging.is_none()
            && !self.kv_offloaded
            && self.qwen3_prepare_fusion
            && self.past_media_embeddings(self.position)
    }

    pub(in crate::inference) fn activate_qwen3_decode(&self, want_logits: bool) -> Option<Scratch> {
        if !self.qwen3_replay_eligible(want_logits) {
            return None;
        }
        self.decode_pool_active.set(true);
        Some(Scratch(self.decode_pool_active.clone()))
    }

    pub(in crate::inference) fn replay_qwen3_decode(
        &mut self,
        token: u32,
        want_logits: bool,
    ) -> Result<bool> {
        let mut saved = self.qwen3_decode_replay.borrow_mut();
        let Some(replay) = saved.as_mut() else {
            return Ok(false);
        };
        let matches = self.qwen3_replay_eligible(want_logits) && self.position == replay.next
            && layout(self, self.position) == replay.layout
            && self.last_hidden.as_ref().is_some_and(|h| h.cu_deviceptr() == replay.hidden)
            && self.states.len() == replay.caches.len()
            && self.states.iter().zip(&replay.caches).all(|(state, &(kp, vp))|
                matches!(state, State::Attention { k, v } if k.cu_deviceptr() == kp && v.cu_deviceptr() == vp));
        if !matches {
            saved.take();
            return Ok(false);
        }
        for (is_token, scalar) in &mut replay.scalars {
            scalar.set(
                &replay.exec,
                if *is_token {
                    token as usize
                } else {
                    self.position
                },
            )?;
        }
        replay.exec.launch(&self.stream)?;
        self.position += 1;
        replay.next = self.position;
        self.graph_stats.borrow_mut()[0] += 1;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires official MiniCPM-o AWQ checkpoint and CUDA"]
    fn dense_replay_preserves_logits_and_pool_isolation() -> Result<()> {
        let model = std::env::var("LIA_MINICPMO45_MODEL")?;
        let mut engine = Engine::load(&model, 256)?;
        anyhow::ensure!(
            engine.qwen3_replay_eligible(false),
            "enable the frozen replay test configuration"
        );
        let tokens: Vec<u32> = (0..132).map(|i| 101 + (i % 31) as u32).collect();
        let checkpoints = [1, 31, 32, 33, 63, 64, 65, 66, 127, 128, 129, 132];
        let mut expected = BTreeMap::new();
        engine.qwen3_replay_enabled = false;
        for &token in &tokens {
            engine.step(token, false)?;
            if checkpoints.contains(&engine.position) {
                expected.insert(engine.position, engine.current_logits()?);
            }
        }
        engine.reset()?;
        engine.qwen3_replay_enabled = true;
        let before = engine.cuda_graph_stats();
        let mut held = Vec::new();
        let mut saved = None;
        let compare = |position: usize, actual: &[f32]| -> Result<()> {
            let reference = &expected[&position];
            anyhow::ensure!(
                actual.len() == reference.len()
                    && actual
                        .iter()
                        .zip(reference)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                "replay logits differ at position {position}"
            );
            Ok(())
        };
        for (i, &token) in tokens.iter().enumerate() {
            engine.step(token, false)?;
            if i == 0 {
                for _ in 0..2 {
                    let mut foreign = engine.zeros(4096)?;
                    foreign.copy_from_host(&engine.stream, &vec![12345.0f32; 4096])?;
                    held.push(foreign);
                }
            }
            if checkpoints.contains(&engine.position) {
                compare(engine.position, &engine.current_logits()?)?;
            }
            if engine.position == 64 {
                saved = Some(engine.snapshot()?);
            }
        }
        let after = engine.cuda_graph_stats();
        anyhow::ensure!(
            after[0] - before[0] == 132 && after[2] - before[2] < 10 && after[1] == before[1],
            "direct replay did not bypass capture/update: {before:?} -> {after:?}"
        );
        engine.restore(&saved.context("missing replay snapshot")?)?;
        anyhow::ensure!(
            engine.qwen3_decode_replay.borrow().is_none(),
            "restore retained stale graph"
        );
        engine.step(tokens[64], false)?;
        compare(65, &engine.current_logits()?)?;
        let host = engine.park_attention()?;
        anyhow::ensure!(
            engine.qwen3_decode_replay.borrow().is_none(),
            "KV offload retained stale graph"
        );
        engine.restore_attention(host)?;
        engine.step(tokens[65], false)?;
        compare(66, &engine.current_logits()?)?;
        for foreign in held {
            anyhow::ensure!(
                foreign
                    .to_host_vec(&engine.stream)?
                    .iter()
                    .all(|v| v.to_bits() == 12345.0f32.to_bits()),
                "ordinary pool overwrote graph or foreign storage"
            );
        }
        engine.reset()?;
        anyhow::ensure!(
            engine.qwen3_decode_replay.borrow().is_none(),
            "reset retained stale graph"
        );
        println!(
            "Qwen3 replay: 132 teacher-forced steps, bitexact logits across partition boundaries; isolated pool, restore and KV offload passed; stats {before:?} -> {after:?}"
        );
        Ok(())
    }
}
