use super::*;
use anyhow::Context;

impl Engine {
    pub(super) fn weight(&self, name: &str) -> Result<&Weight> {
        anyhow::ensure!(
            !self.kv_offloaded || name.starts_with("vision."),
            "attention cache is offloaded; language execution is unavailable"
        );
        self.weights
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor {name}"))
    }
    pub(super) fn floats(&self, name: &str) -> Result<&DeviceBuffer<f32>> {
        self.weight(name)?
            .floats
            .as_ref()
            .ok_or_else(|| anyhow!("expected F32 tensor {name}"))
    }
    pub(super) fn zeros(&self, n: usize) -> Result<Buffer> {
        let recycled = if self.preallocated {
            self.pool
                .borrow_mut()
                .range_mut(n..)
                .find_map(|(&capacity, buffers)| buffers.pop().map(|data| (capacity, data)))
        } else {
            self.pool
                .borrow_mut()
                .get_mut(&n)
                .and_then(|buffers| buffers.pop())
                .map(|data| (n, data))
        };
        let (capacity, data) = match recycled {
            Some(pair) => pair,
            None => {
                anyhow::ensure!(
                    !self.preallocated,
                    "preallocated scratch exhausted: requested {n} f32 elements"
                );
                (
                    n,
                    DeviceBuffer::zeroed(&self.stream, n)
                        .with_context(|| format!("allocating {n} f32 scratch elements"))?,
                )
            }
        };
        debug_assert_eq!(capacity, data.len());
        Buffer::lease(data, n, self.pool.clone())
    }

    pub(super) fn mm(&self, name: &str, input: &DeviceBuffer<f32>) -> Result<Buffer> {
        self.mm_prepared(name, input, false)
    }
    pub(super) fn mm_prepared(
        &self,
        name: &str,
        input: &DeviceBuffer<f32>,
        reuse_input: bool,
    ) -> Result<Buffer> {
        let w = self.weight(name)?;
        if w.shape.len() != 2 || !input.len().is_multiple_of(w.shape[0]) {
            return Err(anyhow!("matrix shape mismatch: {name}"));
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
        let mut output = self.zeros(rows * batch)?;
        // SAFETY: validated matrix layout, matching input and allocated output;
        // every warp has one disjoint output row.
        if w.group.is_some() {
            self.gsq_mm(w, input, reuse_input, &mut output)?;
        } else if batch >= crate::gpu::WIDE_MIN_BATCH
            && let Some(weight_scale) = w.fp8_scale
        {
            let mut work = self.fp8_work.borrow_mut();
            let work = work
                .as_mut()
                .ok_or_else(|| anyhow!("missing FP8 prefill workspace"))?;
            if !reuse_input {
                crate::gpu::fp8::prepare(
                    &self.module,
                    &self.stream,
                    input,
                    &mut work.maxima,
                    &mut work.scales,
                    &mut work.packed,
                )?;
            }
            let dynamic = batch >= crate::gpu::fp8_cache::MIN_BATCH
                && work
                    .weight_tiles
                    .as_ref()
                    .is_some_and(|buffer| buffer.len() >= rows.div_ceil(64) * 64 * cols / 4);
            if dynamic {
                crate::gpu::fp8_cache::pack_into(
                    &self.module,
                    &self.stream,
                    w.kind,
                    cols,
                    rows,
                    weight_scale,
                    &w.data,
                    work.weight_tiles
                        .as_mut()
                        .expect("validated weight workspace"),
                )?;
            }
            crate::gpu::fp8::launch(
                &self.module,
                &self.stream,
                w.kind,
                cols,
                rows,
                batch,
                weight_scale,
                &w.data,
                &work.packed,
                &work.scales,
                &mut output,
                if dynamic {
                    work.weight_tiles.as_ref()
                } else {
                    None
                },
            )?;
        } else if w.kind == 30 && rows == 256 && cols.is_multiple_of(128) {
            // Keep router operands in FP32 across single-token and batched paths.
            anyhow::ensure!(
                rows.checked_mul(cols).and_then(|n| n.checked_mul(2)) == Some(w.data.len()),
                "invalid BF16 router weight extent"
            );
            // SAFETY: complete aligned BF16 rows and 128-column groups. Each
            // full warp preserves the original lane FMA and reduction order.
            unsafe {
                self.module.bf16_router.bf16_router_matvec(
                    &self.stream,
                    LaunchConfig {
                        grid_dim: ((rows * batch * 32).div_ceil(64) as u32, 1, 1),
                        block_dim: (64, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    cols,
                    rows,
                    batch,
                    &w.data,
                    input,
                    output.cu_deviceptr() as *mut f32,
                )?;
            }
        } else if w.kind == 30 && rows <= 96 {
            let parts = cols.div_ceil(256);
            let partial = self.zeros(parts * rows * batch)?;
            // SAFETY: the checked BF16 matrix has exactly two bytes per value;
            // each partition and output row has a disjoint warp producer.
            unsafe {
                self.module.matrix.bf16_partial(
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
                self.module.matrix.bf16_merge(
                    &self.stream,
                    LaunchConfig::for_num_elems((rows * batch * 32) as u32),
                    parts,
                    &partial,
                    output.cu_deviceptr() as *mut f32,
                )?;
            }
        } else if batch > 8 {
            let mut packed = self.prefill_input.borrow_mut();
            if packed
                .as_ref()
                .is_none_or(|buffer| buffer.len() < input.len() / 2)
            {
                anyhow::ensure!(!self.preallocated, "reserved prefill input extent exceeded");
                *packed = Some(DeviceBuffer::zeroed(&self.stream, input.len() / 2)?);
            }
            let packed = packed
                .as_ref()
                .ok_or_else(|| anyhow!("missing prefill input workspace"))?;
            // SAFETY: checked matrices and workspace on one stream. Reuse is
            // confined to mm_many's identical immutable input and shared width.
            unsafe {
                crate::gpu::launch_prefill_matrix(
                    &self.module,
                    &self.stream,
                    w.kind,
                    cols,
                    rows,
                    batch,
                    &w.data,
                    input,
                    reuse_input,
                    packed,
                    output.cu_deviceptr() as *mut f32,
                    self.prefill_tile,
                )?;
            }
        } else if self.integer && w.kind != 0 && w.kind != 30 {
            let mut work = self.quant_work.borrow_mut();
            if let std::collections::btree_map::Entry::Vacant(entry) = work.entry(input.len()) {
                anyhow::ensure!(
                    !self.preallocated,
                    "unreserved activation quantization extent"
                );
                entry.insert(QuantWork {
                    packed: DeviceBuffer::zeroed(&self.stream, input.len() / 4)?,
                    scales: DeviceBuffer::zeroed(&self.stream, input.len() / 32)?,
                });
            }
            let work = work
                .get(&input.len())
                .ok_or_else(|| anyhow!("missing quantization workspace"))?;
            // SAFETY: complete 32-element activation groups, correctly sized
            // persistent workspaces, and a disjoint output row per warp. Every
            // use of this workspace is ordered on the same stream.
            unsafe {
                if !reuse_input {
                    self.module.integer.quantize_activation(
                        &self.stream,
                        LaunchConfig::for_num_elems(input.len() as u32),
                        input,
                        work.packed.cu_deviceptr() as *mut u32,
                        work.scales.cu_deviceptr() as *mut f32,
                    )?;
                }
                crate::gpu::launch_int_matmul(
                    &self.module,
                    &self.stream,
                    w.kind,
                    cols,
                    rows,
                    batch,
                    self.split_k,
                    self.row_group,
                    self.matrix_threads,
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
    pub(super) fn norm(&self, name: &str, input: &DeviceBuffer<f32>, dim: usize) -> Result<Buffer> {
        let weight = self.floats(name)?;
        if !input.len().is_multiple_of(dim) || weight.len() != dim {
            return Err(anyhow!("norm shape mismatch: {name}"));
        }
        let groups = input.len() / dim;
        let output = self.zeros(input.len())?;
        // SAFETY: complete disjoint groups, with matching normalization weights.
        unsafe {
            if dim >= 512 {
                self.module.normalization.rms_norm_block(
                    &self.stream,
                    LaunchConfig {
                        grid_dim: (groups as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    dim,
                    groups,
                    self.eps,
                    input,
                    weight,
                    output.cu_deviceptr() as *mut f32,
                )?;
            } else {
                self.module.normalization.rms_norm(
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
        }
        Ok(output)
    }
    pub(super) fn add(&self, a: &DeviceBuffer<f32>, b: &DeviceBuffer<f32>) -> Result<Buffer> {
        if a.len() != b.len() {
            return Err(anyhow!("residual length mismatch"));
        }
        let mut out = self.zeros(a.len())?;
        // SAFETY: equally sized buffers and guarded element-wise output.
        unsafe {
            self.module.basic.residual(
                &self.stream,
                LaunchConfig::for_num_elems(a.len() as u32),
                a,
                b,
                &mut out,
            )?;
        }
        Ok(out)
    }
    pub(super) fn silu_mul(
        &self,
        x: &DeviceBuffer<f32>,
        gate: &DeviceBuffer<f32>,
    ) -> Result<Buffer> {
        if x.len() != gate.len() {
            return Err(anyhow!("gate length mismatch"));
        }
        let mut out = self.zeros(x.len())?;
        // SAFETY: equally sized buffers and guarded element-wise output.
        unsafe {
            self.module.basic.silu_mul(
                &self.stream,
                LaunchConfig::for_num_elems(x.len() as u32),
                x,
                gate,
                &mut out,
            )?;
        }
        Ok(out)
    }
}
