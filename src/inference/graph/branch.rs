//! Fork independent captured work without changing numerical operators.
use super::*;

pub(in crate::inference) struct Frontier {
    nodes: Vec<sys::CUgraphNode>,
    edges: Vec<sys::CUgraphEdgeData>,
}

impl Frontier {
    pub(in crate::inference) fn current(stream: &CudaStream) -> Result<Option<Self>> {
        let mut status = 0;
        let mut nodes = std::ptr::null();
        let mut edges = std::ptr::null();
        let mut count = 0;
        // SAFETY: the current thread owns this stream's capture. The driver
        // arrays are copied before another capture operation can invalidate them.
        check(
            unsafe {
                sys::cuStreamGetCaptureInfo_v3(
                    stream.cu_stream(),
                    &mut status,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &mut nodes,
                    &mut edges,
                    &mut count,
                )
            },
            "reading capture frontier",
        )?;
        if status != sys::CUstreamCaptureStatus_enum_CU_STREAM_CAPTURE_STATUS_ACTIVE {
            return Ok(None);
        }
        anyhow::ensure!(
            count > 0 && !nodes.is_null() && !edges.is_null(),
            "empty capture frontier"
        );
        // SAFETY: a nonempty active frontier has count live node/edge entries.
        Ok(Some(unsafe {
            Self {
                nodes: std::slice::from_raw_parts(nodes, count).to_vec(),
                edges: std::slice::from_raw_parts(edges, count).to_vec(),
            }
        }))
    }

    pub(in crate::inference) fn restore(&mut self, stream: &CudaStream) -> Result<()> {
        self.update(
            stream,
            sys::CUstreamUpdateCaptureDependencies_flags_enum_CU_STREAM_SET_CAPTURE_DEPENDENCIES,
        )
    }

    pub(in crate::inference) fn join(&mut self, stream: &CudaStream) -> Result<()> {
        self.update(
            stream,
            sys::CUstreamUpdateCaptureDependencies_flags_enum_CU_STREAM_ADD_CAPTURE_DEPENDENCIES,
        )
    }

    fn update(&mut self, stream: &CudaStream, flags: u32) -> Result<()> {
        // SAFETY: the frontier belongs to this still-active capture. All branch
        // endpoints must be joined before consumers or scratch reuse are queued.
        check(
            unsafe {
                sys::cuStreamUpdateCaptureDependencies_v2(
                    stream.cu_stream(),
                    self.nodes.as_mut_ptr(),
                    self.edges.as_ptr(),
                    self.nodes.len(),
                    flags,
                )
            },
            "updating capture frontier",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_branches_join_before_consumers_and_replay() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.new_stream()?;
        let module = kernels::load(&context)?;
        let mut a = DeviceBuffer::from_host(&stream, &[1.0f32; 1024])?;
        let b = DeviceBuffer::from_host(&stream, &[2.0f32; 1024])?;
        let mut root = DeviceBuffer::zeroed(&stream, 1024)?;
        let mut left = DeviceBuffer::zeroed(&stream, 1024)?;
        let mut right = DeviceBuffer::zeroed(&stream, 1024)?;
        let mut output = DeviceBuffer::zeroed(&stream, 1024)?;
        stream.synchronize()?;
        assert!(Frontier::current(&stream)?.is_none());
        let capture = Capture::begin(stream.clone())?;
        let config = LaunchConfig::for_num_elems(1024);
        // SAFETY: every array has 1024 elements and each output is disjoint.
        unsafe { module.basic.residual(&stream, config, &a, &b, &mut root)? };
        let mut fork = Frontier::current(&stream)?.context("missing root frontier")?;
        // SAFETY: the root precedes this independent branch; inputs stay live.
        unsafe {
            module
                .basic
                .residual(&stream, config, &root, &a, &mut left)?
        };
        let mut left_end = Frontier::current(&stream)?.context("missing left frontier")?;
        fork.restore(&stream)?;
        // SAFETY: this branch also depends on root, and writes separate storage.
        unsafe {
            module
                .basic
                .residual(&stream, config, &root, &b, &mut right)?
        };
        left_end.join(&stream)?;
        let join = Frontier::current(&stream)?.context("missing joined frontier")?;
        assert_eq!(join.nodes.len(), 2);
        // SAFETY: both producers are explicit dependencies of this consumer.
        unsafe {
            module
                .basic
                .residual(&stream, config, &left, &right, &mut output)?
        };
        let graph = capture.end()?;
        let executable = Executable::instantiate(&graph, context)?;
        for value in [1.0f32, 3.0, -5.0] {
            // Replaying with changed data checks the edges on every execution.
            a.copy_from_host(&stream, &[value; 1024])?;
            executable.launch(&stream)?;
            assert_eq!(output.to_host_vec(&stream)?, vec![3.0 * value + 6.0; 1024]);
        }
        Ok(())
    }
}
