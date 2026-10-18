//! Replay a fixed non-speculative decode schedule between attention tile boundaries.
use super::*;
use std::ffi::CStr;

struct Parameter {
    node: sys::CUgraphNode,
    params: sys::CUDA_KERNEL_NODE_PARAMS_v2,
    index: usize,
    value: Value,
    pointers: Vec<*mut std::ffi::c_void>,
}

enum Value {
    Token,
    Position,
    Tokens,
}

pub(in crate::inference) struct Replay {
    pub(super) exec: Executable,
    // The immutable graph owns its argument backing memory. Preallocation keeps
    // every captured scratch allocation in the engine pool for the replay lifetime.
    _graph: Graph,
    parameters: Vec<Parameter>,
    hidden: u64,
    next: usize,
    layout: attention_plan::Plan,
}

impl Replay {
    pub(super) fn new(
        graph: Graph,
        context: Arc<CudaContext>,
        hidden: u64,
        position: usize,
        layout: attention_plan::Plan,
    ) -> Result<Self> {
        let mut count = 0;
        // SAFETY: the graph is a completed capture, retained by the returned owner.
        check(
            unsafe { sys::cuGraphGetNodes(graph.0, std::ptr::null_mut(), &mut count) },
            "counting decode nodes",
        )?;
        let mut nodes = vec![std::ptr::null_mut(); count];
        check(
            unsafe { sys::cuGraphGetNodes(graph.0, nodes.as_mut_ptr(), &mut count) },
            "reading decode nodes",
        )?;
        let mut parameters = Vec::new();
        let mut embedding = 0;
        let mut rotary = 0;
        let mut caches = 0;
        let mut attention = 0;
        for node in nodes {
            let mut kind = 0;
            check(
                unsafe { sys::cuGraphNodeGetType(node, &mut kind) },
                "reading decode node kind",
            )?;
            anyhow::ensure!(
                kind == sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL,
                "decode replay requires kernel-only capture"
            );
            let mut params: sys::CUDA_KERNEL_NODE_PARAMS_v2 = unsafe { std::mem::zeroed() };
            check(
                unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut params) },
                "reading decode kernel",
            )?;
            let mut name = std::ptr::null();
            check(
                unsafe { sys::cuFuncGetName(&mut name, params.func) },
                "reading decode kernel name",
            )?;
            let name = unsafe { CStr::from_ptr(name) }.to_str()?;
            let change = if name == "embedding" {
                embedding += 1;
                Some((4, Value::Token))
            } else if name == "table" {
                rotary += 1;
                Some((1, Value::Position))
            } else if name == "rope_table" {
                rotary += 1;
                Some((2, Value::Position))
            } else if name == "cache_kv_half" {
                caches += 1;
                Some((1, Value::Position))
            } else if name.starts_with("flash_partial_half_") || name.starts_with("flash_mma_") {
                attention += 1;
                Some((2, Value::Tokens))
            } else {
                None
            };
            if let Some((index, value)) = change {
                let (mut offset, mut size) = (0, 0);
                check(
                    unsafe { sys::cuFuncGetParamInfo(params.func, index, &mut offset, &mut size) },
                    "checking decode scalar ABI",
                )?;
                anyhow::ensure!(
                    size == std::mem::size_of::<usize>()
                        && !params.kernelParams.is_null()
                        && params.extra.is_null(),
                    "unsupported decode scalar ABI"
                );
                let mut pointers = Vec::new();
                loop {
                    let status = unsafe {
                        sys::cuFuncGetParamInfo(params.func, pointers.len(), &mut offset, &mut size)
                    };
                    if status == sys::cudaError_enum_CUDA_ERROR_INVALID_VALUE {
                        break;
                    }
                    check(status, "reading decode parameter count")?;
                    pointers.push(unsafe { *params.kernelParams.add(pointers.len()) });
                }
                let original = unsafe { pointers[index].cast::<usize>().read_unaligned() };
                let expected = match value {
                    Value::Token => original,
                    Value::Position => position,
                    Value::Tokens => position + 1,
                };
                anyhow::ensure!(
                    original == expected,
                    "decode scalar parameter mismatch for {name}"
                );
                parameters.push(Parameter {
                    node,
                    params,
                    index,
                    value,
                    pointers,
                });
            }
        }
        anyhow::ensure!(
            embedding == 1 && rotary == 1 && caches == 10 && attention == 10,
            "unexpected decode topology: embedding={embedding}, rotary={rotary}, caches={caches}, attention={attention}"
        );
        let exec = Executable::instantiate(&graph, context)?;
        Ok(Self {
            exec,
            _graph: graph,
            parameters,
            hidden,
            next: position + 1,
            layout,
        })
    }

    fn launch(&mut self, stream: &CudaStream, token: u32, position: usize) -> Result<()> {
        for parameter in &mut self.parameters {
            let value = match parameter.value {
                Value::Token => token as usize,
                Value::Position => position,
                Value::Tokens => position + 1,
            };
            // Driver-owned argument memory remains immutable. A temporary pointer
            // array replaces only the validated scalar; the driver copies it here.
            parameter.pointers[parameter.index] = (&value as *const usize).cast_mut().cast();
            let mut params = parameter.params;
            params.kernelParams = parameter.pointers.as_mut_ptr();
            check(
                unsafe {
                    sys::cuGraphExecKernelNodeSetParams_v2(
                        self.exec.handle,
                        parameter.node,
                        &params,
                    )
                },
                "updating decode scalar",
            )?;
        }
        self.exec.launch(stream)?;
        self.next = position + 1;
        Ok(())
    }
}

impl Engine {
    pub(in crate::inference) fn replay_decode(
        &mut self,
        token: u32,
        want_logits: bool,
    ) -> Result<bool> {
        let mut saved = self.decode_replay.borrow_mut();
        let eligible = !want_logits && !self.capture && self.geometry.moe && !self.geometry.mtp;
        let Some(replay) = saved.as_mut() else {
            return Ok(false);
        };
        let matches = eligible
            && self.position == replay.next
            && attention_plan::plan(
                1,
                self.position + 1,
                self.attention_mma_decode,
                self.attention_query_tiles,
            ) == replay.layout
            && self
                .last_hidden
                .as_ref()
                .is_some_and(|h| h.cu_deviceptr() == replay.hidden);
        if !matches {
            saved.take();
            return Ok(false);
        }
        replay.launch(&self.stream, token, self.position)?;
        self.rotary.borrow_mut().key = Some((self.position, 1));
        self.position += 1;
        self.graph_stats.borrow_mut()[0] += 1;
        Ok(true)
    }
}
