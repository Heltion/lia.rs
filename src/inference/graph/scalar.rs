//! A checked scalar argument update retaining the captured graph's other arguments.
use super::*;

pub(super) struct Scalar {
    node: sys::CUgraphNode,
    params: sys::CUDA_KERNEL_NODE_PARAMS_v2,
    pointers: Vec<*mut std::ffi::c_void>,
    index: usize,
}

impl Scalar {
    pub(super) fn new(
        node: sys::CUgraphNode,
        params: sys::CUDA_KERNEL_NODE_PARAMS_v2,
        index: usize,
        expected: Option<usize>,
    ) -> Result<Self> {
        let (mut offset, mut size) = (0, 0);
        // SAFETY: caller obtained this function from a live captured kernel node.
        check(
            unsafe { sys::cuFuncGetParamInfo(params.func, index, &mut offset, &mut size) },
            "checking Qwen3 graph scalar ABI",
        )?;
        anyhow::ensure!(
            size == std::mem::size_of::<usize>()
                && !params.kernelParams.is_null()
                && params.extra.is_null(),
            "unsupported Qwen3 graph scalar ABI"
        );
        let mut pointers = Vec::new();
        for i in 0..64 {
            // SAFETY: the driver validates each queried argument index.
            let status = unsafe { sys::cuFuncGetParamInfo(params.func, i, &mut offset, &mut size) };
            if status == sys::cudaError_enum_CUDA_ERROR_INVALID_VALUE {
                break;
            }
            check(status, "reading Qwen3 graph argument count")?;
            // SAFETY: the successful query established an existing argument.
            pointers.push(unsafe { *params.kernelParams.add(i) });
        }
        anyhow::ensure!(
            index < pointers.len() && pointers.len() < 64,
            "invalid Qwen3 graph argument table"
        );
        if let Some(expected) = expected {
            // SAFETY: the queried scalar has exactly usize size; alignment is not assumed.
            let original = unsafe { pointers[index].cast::<usize>().read_unaligned() };
            anyhow::ensure!(original == expected, "Qwen3 captured position mismatch");
        }
        Ok(Self {
            node,
            params,
            pointers,
            index,
        })
    }

    pub(super) fn set(&mut self, exec: &Executable, value: usize) -> Result<()> {
        self.pointers[self.index] = (&value as *const usize).cast_mut().cast();
        let mut params = self.params;
        params.kernelParams = self.pointers.as_mut_ptr();
        // SAFETY: the immutable graph retains all other argument backing storage;
        // this synchronous driver call copies the replacement scalar before return.
        check(
            unsafe { sys::cuGraphExecKernelNodeSetParams_v2(exec.handle, self.node, &params) },
            "updating Qwen3 graph scalar",
        )
    }
}
