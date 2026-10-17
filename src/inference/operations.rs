use super::*;

impl Engine {
    pub(super) fn weight(&self, name: &str) -> Result<&Weight> {
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
        let output = self.zeros(rows * batch)?;
        // SAFETY: validated matrix layout, matching input and allocated output;
        // every warp has one disjoint output row.
        if batch > 8 {
            let mut packed = self.prefill_input.borrow_mut();
            if packed
                .as_ref()
                .is_none_or(|buffer| buffer.len() < input.len() / 2)
            {
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
