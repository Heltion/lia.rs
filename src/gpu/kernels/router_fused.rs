//! Exact BF16 router GEMV followed by selection in the last completing CTA.
use cuda_device::{
    atomic::{AtomicOrdering, DeviceAtomicU32},
    kernel, launch_bounds, launch_contract, thread,
};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::super::moe_route::select;
    use super::*;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain=1,block=(256,1,1))]
    pub unsafe fn router_fused(
        columns: usize,
        batch: usize,
        weights: &[u8],
        input: &[f32],
        logits: *mut f32,
        counters: *mut u32,
        selected: *mut u32,
        probabilities: *mut f32,
    ) {
        static mut LAST: cuda_device::SharedArray<u32, 1, 4> = cuda_device::SharedArray::UNINIT;
        let tid = thread::index_1d().get();
        let local = tid % 256;
        let lane = local % 32;
        let block = tid / 256;
        let token = block / 32;
        if token >= batch {
            return;
        }
        let row = block % 32 * 8 + local / 32;
        let mut column = lane;
        let mut sum = 0.0f32;
        while column < columns {
            let mut values = [0.0f32; 4];
            let mut x = [0.0f32; 4];
            let mut i = 0;
            #[unroll]
            while i < 4 {
                let col = column + i * 32;
                let word = unsafe {
                    weights
                        .as_ptr()
                        .cast::<u16>()
                        .add(row * columns + col)
                        .read()
                };
                values[i] = f32::from_bits((word as u32) << 16);
                x[i] = unsafe { *input.get_unchecked(token * columns + col) };
                i += 1;
            }
            i = 0;
            #[unroll]
            while i < 4 {
                sum = values[i].mul_add(x[i], sum);
                i += 1;
            }
            column += 128;
        }
        let mut delta = 16;
        while delta > 0 {
            sum += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum, delta);
            delta /= 2;
        }
        if lane == 0 {
            unsafe { *logits.add(token * 256 + row) = sum };
        }
        // Each CTA publishes all eight warp stores before its release operation.
        // The acquire/release RMW chain makes all 256 logits visible to the
        // last CTA. No CTA waits for another, so occupancy cannot deadlock it.
        thread::sync_threads();
        let last = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut LAST) };
        if local == 0 {
            let counter = unsafe { DeviceAtomicU32::from_ptr(counters.add(token)) };
            unsafe { *last = u32::from(counter.fetch_add(1, AtomicOrdering::AcqRel) == 31) };
        }
        thread::sync_threads();
        if unsafe { *last } != 0 && local < 32 {
            let mut values = [0.0f32; 8];
            let mut j = 0;
            while j < 8 {
                // Coherent global reads are required: these stores came from
                // other CTAs and must not use a read-only noncoherent cache.
                values[j] = unsafe { logits.add(token * 256 + lane + j * 32).read_volatile() };
                j += 1;
            }
            select!(values, lane, token, selected, probabilities);
            if lane == 0 {
                // Subsequent invocations are ordered on this same stream;
                // every token starts again at zero, including graph replays.
                unsafe { DeviceAtomicU32::from_ptr(counters.add(token)) }
                    .store(0, AtomicOrdering::Release);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Result, ensure};
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig, sys};

    struct Captured {
        graph: sys::CUgraph,
        exec: sys::CUgraphExec,
    }
    impl Drop for Captured {
        fn drop(&mut self) {
            // SAFETY: the test owns both handles and synchronizes every replay.
            unsafe {
                sys::cuGraphExecDestroy(self.exec);
                sys::cuGraphDestroy(self.graph);
            }
        }
    }

    #[test]
    fn fused_router_preserves_bits_guards_and_changed_input_graph_replays() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.new_stream()?;
        let module = super::super::load(&context)?;
        for (columns, batch) in [(128usize, 1usize), (2048, 1), (512, 3), (2048, 8)] {
            let weights = (0..256 * columns)
                .flat_map(|i| {
                    let v = ((i * 73 % 257) as f32 - 128.0) / 101.0;
                    ((v.to_bits() >> 16) as u16).to_le_bytes()
                })
                .collect::<Vec<_>>();
            let weights = DeviceBuffer::from_host(&stream, &weights)?;
            let mut input = DeviceBuffer::zeroed(&stream, columns * batch)?;
            let mut counts = vec![0u32; batch];
            counts.extend([0x1234_5678; 17]);
            let counts = DeviceBuffer::from_host(&stream, &counts)?;
            let guard = f32::from_bits(0x3f12_3456);
            let logits = DeviceBuffer::from_host(&stream, &vec![guard; batch * 256 + 17])?;
            let ids = DeviceBuffer::from_host(&stream, &vec![0x1234_5678u32; batch * 8 + 17])?;
            let probs = DeviceBuffer::from_host(&stream, &vec![guard; batch * 8 + 17])?;
            let reference = DeviceBuffer::<f32>::zeroed(&stream, batch * 256)?;
            let reference_ids = DeviceBuffer::<u32>::zeroed(&stream, batch * 8)?;
            let reference_probs = DeviceBuffer::<f32>::zeroed(&stream, batch * 8)?;
            let plan =
                module
                    .router_fused
                    .prepare_router_fused(cuda_core::simt::LaunchConfig1D::new(
                        (batch * 32) as u32,
                        256,
                        0,
                    ))?;
            stream.synchronize()?;
            let mut graph = std::ptr::null_mut();
            // SAFETY: this nonblocking stream has no active capture; all
            // buffers and prepared launch arguments remain live below.
            ensure!(
                unsafe {
                    sys::cuStreamBeginCapture_v2(
                        stream.cu_stream(),
                        sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                    )
                } == 0,
                "beginning fused router test capture"
            );
            let recorded = unsafe {
                module.router_fused.router_fused(
                    &stream,
                    &plan,
                    columns,
                    batch,
                    &weights,
                    &input,
                    logits.cu_deviceptr() as *mut f32,
                    counts.cu_deviceptr() as *mut u32,
                    ids.cu_deviceptr() as *mut u32,
                    probs.cu_deviceptr() as *mut f32,
                )
            };
            let ended = unsafe { sys::cuStreamEndCapture(stream.cu_stream(), &mut graph) };
            recorded?;
            ensure!(
                ended == 0 && !graph.is_null(),
                "ending fused router test capture"
            );
            let mut exec = std::ptr::null_mut();
            ensure!(
                unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0) } == 0,
                "instantiating fused router test graph"
            );
            let captured = Captured { graph, exec };
            for step in 0..7 {
                let values = (0..columns * batch)
                    .map(|i| {
                        if step == 1 {
                            0.0
                        } else {
                            ((i * 37 + step * 19) % 997) as f32 / 71.0 - 7.0
                        }
                    })
                    .collect::<Vec<_>>();
                input.copy_from_host(&stream, &values)?;
                // SAFETY: both reference launches cover complete rows/warps;
                // every replay shares the same live, stream-ordered buffers.
                unsafe {
                    module.bf16_router.bf16_router_matvec(
                        &stream,
                        LaunchConfig {
                            grid_dim: ((batch * 256 * 32).div_ceil(64) as u32, 1, 1),
                            block_dim: (64, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        columns,
                        256,
                        batch,
                        &weights,
                        &input,
                        reference.cu_deviceptr() as *mut f32,
                    )?;
                    module.moe.route(
                        &stream,
                        LaunchConfig::for_num_elems((batch * 32) as u32),
                        &reference,
                        reference_ids.cu_deviceptr() as *mut u32,
                        reference_probs.cu_deviceptr() as *mut f32,
                    )?;
                    for _ in 0..3 {
                        ensure!(
                            sys::cuGraphLaunch(captured.exec, stream.cu_stream()) == 0,
                            "replaying fused router test graph"
                        );
                    }
                }
                let actual = logits.to_host_vec(&stream)?;
                assert_eq!(
                    actual[..batch * 256]
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>(),
                    reference
                        .to_host_vec(&stream)?
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>(),
                    "logits: columns={columns} batch={batch} step={step}"
                );
                assert!(
                    actual[batch * 256..]
                        .iter()
                        .all(|x| x.to_bits() == guard.to_bits())
                );
                let actual_ids = ids.to_host_vec(&stream)?;
                assert_eq!(
                    &actual_ids[..batch * 8],
                    reference_ids.to_host_vec(&stream)?
                );
                assert!(actual_ids[batch * 8..].iter().all(|&x| x == 0x1234_5678));
                let actual = probs.to_host_vec(&stream)?;
                assert_eq!(
                    actual[..batch * 8]
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>(),
                    reference_probs
                        .to_host_vec(&stream)?
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>()
                );
                assert!(
                    actual[batch * 8..]
                        .iter()
                        .all(|x| x.to_bits() == guard.to_bits())
                );
                let counts = counts.to_host_vec(&stream)?;
                assert!(counts[..batch].iter().all(|&x| x == 0));
                assert!(counts[batch..].iter().all(|&x| x == 0x1234_5678));
                if step == 1 {
                    for token in 0..batch {
                        assert_eq!(&actual_ids[token * 8..][..8], &[0, 1, 2, 3, 4, 5, 6, 7]);
                    }
                }
            }
        }
        Ok(())
    }
}
