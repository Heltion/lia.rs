//! Capture GPU-only work and retain bounded decode graphs for repeated launches.
use super::*;
use anyhow::Context;
use cuda_core::sys;
pub(super) mod replay;

fn check(status: sys::CUresult, operation: &str) -> Result<()> {
    anyhow::ensure!(
        status == 0,
        "{operation}: {:?}",
        cuda_core::DriverError(status)
    );
    Ok(())
}

pub(super) struct Capture {
    stream: Arc<CudaStream>,
    active: bool,
}
struct Graph(sys::CUgraph);
impl Drop for Graph {
    fn drop(&mut self) {
        // SAFETY: this object uniquely owns a completed capture's graph.
        unsafe { sys::cuGraphDestroy(self.0) };
    }
}
impl Capture {
    fn begin(stream: Arc<CudaStream>) -> Result<Self> {
        // SAFETY: inference uses a dedicated non-blocking stream and never nests captures.
        check(
            unsafe {
                sys::cuStreamBeginCapture_v2(
                    stream.cu_stream(),
                    sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )
            },
            "beginning CUDA capture",
        )?;
        Ok(Self {
            stream,
            active: true,
        })
    }
    fn end(mut self) -> Result<Graph> {
        let mut graph = std::ptr::null_mut();
        // SAFETY: this guard owns the matching active capture.
        let status = unsafe { sys::cuStreamEndCapture(self.stream.cu_stream(), &mut graph) };
        self.active = false;
        check(status, "ending CUDA capture")?;
        anyhow::ensure!(!graph.is_null(), "CUDA capture returned a null graph");
        Ok(Graph(graph))
    }
}
impl Drop for Capture {
    fn drop(&mut self) {
        if self.active {
            let mut graph = std::ptr::null_mut();
            // SAFETY: unwind/error cleanup terminates the capture without submitting it.
            unsafe { sys::cuStreamEndCapture(self.stream.cu_stream(), &mut graph) };
            if !graph.is_null() {
                drop(Graph(graph));
            }
        }
    }
}

pub(super) struct Executable {
    handle: sys::CUgraphExec,
    _context: Arc<CudaContext>,
}
impl Executable {
    fn instantiate(graph: &Graph, context: Arc<CudaContext>) -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        // SAFETY: the completed graph remains alive through instantiation.
        check(
            unsafe { sys::cuGraphInstantiateWithFlags(&mut handle, graph.0, 0) },
            "instantiating CUDA graph",
        )?;
        Ok(Self {
            handle,
            _context: context,
        })
    }
    fn update(&mut self, graph: &Graph) -> Result<bool> {
        // SAFETY: zero is a valid initial value for this driver output structure.
        let mut info: sys::CUgraphExecUpdateResultInfo = unsafe { std::mem::zeroed() };
        // SAFETY: both handles are live; the driver reports topology incompatibility.
        let status = unsafe { sys::cuGraphExecUpdate_v2(self.handle, graph.0, &mut info) };
        if status == 0
            && info.result == sys::CUgraphExecUpdateResult_enum_CU_GRAPH_EXEC_UPDATE_SUCCESS
        {
            Ok(true)
        } else if status == sys::cudaError_enum_CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE {
            Ok(false)
        } else {
            check(status, "updating CUDA graph")?;
            Ok(false)
        }
    }
    fn launch(&self, stream: &CudaStream) -> Result<()> {
        // SAFETY: input/output allocations remain owned by the engine or its scratch pool,
        // and subsequent reuse is ordered on this same stream.
        check(
            unsafe { sys::cuGraphLaunch(self.handle, stream.cu_stream()) },
            "launching CUDA graph",
        )
    }
}
impl Drop for Executable {
    fn drop(&mut self) {
        // SAFETY: the executable is uniquely owned; CUDA retains in-flight launch resources.
        unsafe { sys::cuGraphExecDestroy(self.handle) };
    }
}

impl Engine {
    pub(super) fn begin_graph(&self) -> Result<Option<Capture>> {
        self.cuda_graph
            .then(|| Capture::begin(self.stream.clone()))
            .transpose()
    }
    pub(super) fn finish_graph<T>(
        &self,
        capture: Option<Capture>,
        key: (u8, usize, bool, bool),
        result: Result<T>,
    ) -> Result<T> {
        let Some(capture) = capture else {
            return result;
        };
        // End capture even when an operator failed, leaving the stream usable for reset.
        let graph = capture.end();
        let value = result?;
        let graph = graph.context("capturing GPU forward execution")?;
        if key == (1, 1, false, false)
            && self.geometry.moe
            && !self.geometry.mtp
            && self.preallocated
            && self.past_image_embeddings(self.position - 1)
            && std::env::var("LIA_DECODE_REPLAY").as_deref() != Ok("0")
            && self
                .rotary
                .borrow()
                .plan
                .as_ref()
                .is_none_or(|p| self.position > p.len())
        {
            let hidden = self
                .last_hidden
                .as_ref()
                .context("missing replay hidden state")?
                .cu_deviceptr();
            let replay = replay::Replay::new(
                graph,
                self._context.clone(),
                hidden,
                self.position - 1,
                attention_plan::plan(
                    1,
                    self.position,
                    self.attention_mma_decode,
                    self.attention_query_tiles,
                ),
            )?;
            replay.exec.launch(&self.stream)?;
            *self.decode_replay.borrow_mut() = Some(replay);
            let mut stats = self.graph_stats.borrow_mut();
            stats[0] += 1;
            stats[2] += 1;
            return Ok(value);
        }
        let mut cache = self.graphs.borrow_mut();
        let updated = match cache.get_mut(&key) {
            Some(exec) => exec.update(&graph)?,
            None => false,
        };
        if !updated {
            cache.insert(key, Executable::instantiate(&graph, self._context.clone())?);
        }
        cache
            .get(&key)
            .expect("instantiated CUDA graph")
            .launch(&self.stream)?;
        let mut stats = self.graph_stats.borrow_mut();
        stats[0] += 1;
        stats[if updated { 1 } else { 2 }] += 1;
        Ok(value)
    }
    pub fn cuda_graph_enabled(&self) -> bool {
        self.cuda_graph
    }
    /// Graph launches, executable updates, and executable instantiations.
    pub fn cuda_graph_stats(&self) -> [u64; 3] {
        *self.graph_stats.borrow()
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // Drain launches before destroying executable graphs and their backing allocations.
        let _ = self.stream.synchronize();
        self.graphs.get_mut().clear();
    }
}
