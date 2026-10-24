//! Packed requests share matrix execution while retaining separate logical KV tables.
use super::*;
use anyhow::{Context, ensure};
pub const PAGE_TOKENS: usize = 32;

#[derive(Clone, Copy, Debug)]
pub struct PagedRow {
    pub token: u32,
    pub position: u32,
    pub sequence: u32,
}

pub(super) struct Work {
    pub positions: DeviceBuffer<u32>,
    pub coefficients: DeviceBuffer<f32>,
    pub sequences: DeviceBuffer<u32>,
    pub pages: DeviceBuffer<u32>,
    pub stride: usize,
    pub parts: usize,
    pub span: usize,
    pub groups: Vec<super::llama_plan::Group>,
    context: usize,
    maximum_sequences: usize,
    outputs: DeviceBuffer<u32>,
    pub upload: Option<super::paging_upload::Upload>,
}

impl Engine {
    /// Reserve metadata for 32-token pages. Engine capacity is the physical token pool.
    pub fn configure_paging(&mut self, maximum_sequences: usize, context: usize) -> Result<()> {
        ensure!(
            self.geometry.llama,
            "paged execution currently requires MiniCPM5"
        );
        ensure!(
            self.position == 0 && self.paging.is_none(),
            "paging must be configured before execution"
        );
        ensure!(
            (1..=64).contains(&maximum_sequences),
            "invalid sequence count"
        );
        ensure!(
            (1..=131072).contains(&context),
            "invalid per-request context"
        );
        ensure!(
            self.capacity.is_multiple_of(PAGE_TOKENS),
            "physical KV pool must contain complete pages"
        );
        if self.preallocated && maximum_sequences > 8 {
            let elements = maximum_sequences * self.vocabulary_size();
            let mut pool = self.pool.borrow_mut();
            let buffers = pool.entry(elements).or_default();
            for _ in 0..2 {
                buffers.push(
                    DeviceBuffer::zeroed(&self.stream, elements)
                        .context("reserving large-batch vocabulary outputs")?,
                );
            }
            self.scratch_reserved += 2 * elements * 4;
        }
        let stride = context.div_ceil(PAGE_TOKENS);
        let rows = crate::gpu::MAX_PREFILL_BATCH;
        *self.nucleus.borrow_mut() = Some(crate::gpu::nucleus::Workspace::new(
            &self.stream,
            self.vocabulary_size(),
            maximum_sequences,
        )?);
        self.paging = Some(Work {
            positions: DeviceBuffer::zeroed(&self.stream, rows)?,
            coefficients: DeviceBuffer::zeroed(&self.stream, rows * 128)?,
            sequences: DeviceBuffer::zeroed(&self.stream, rows)?,
            pages: DeviceBuffer::zeroed(&self.stream, maximum_sequences * stride)?,
            outputs: DeviceBuffer::zeroed(&self.stream, maximum_sequences)?,
            upload: if std::env::var("LIA_KV_PINNED_UPLOAD").as_deref() != Ok("0") {
                Some(super::paging_upload::Upload::new(
                    &self._context,
                    &self.stream,
                )?)
            } else {
                None
            },
            context,
            maximum_sequences,
            stride,
            parts: 1,
            span: 256,
            groups: Vec::new(),
        });
        self.stream.synchronize()?;
        Ok(())
    }

    /// Execute packed causal rows. Only requested rows project to vocabulary logits.
    /// Tables must refer exclusively to owned pages or immutable shared prefixes.
    pub fn forward_paged(
        &mut self,
        rows: &[PagedRow],
        tables: &[Vec<u32>],
        outputs: &[u32],
    ) -> Result<Vec<f32>> {
        self.run_paged(rows, tables, outputs, |engine, logits| {
            logits
                .map(|v| v.to_host_vec(&engine.stream).map_err(anyhow::Error::from))
                .transpose()
                .map(Option::unwrap_or_default)
        })
    }

    pub(super) fn run_paged<T>(
        &mut self,
        rows: &[PagedRow],
        tables: &[Vec<u32>],
        outputs: &[u32],
        receive: impl FnOnce(&Self, Option<&DeviceBuffer<f32>>) -> Result<T>,
    ) -> Result<T> {
        let vocabulary = self.vocabulary_size();
        let work = self.paging.as_mut().context("paging is not configured")?;
        ensure!(
            !rows.is_empty() && rows.len() <= self.prefill_batch,
            "invalid packed row count"
        );
        ensure!(
            outputs.len() <= work.maximum_sequences,
            "invalid output row count"
        );
        ensure!(
            outputs.iter().all(|&i| (i as usize) < rows.len()),
            "output row outside packed input"
        );
        ensure!(
            tables.len() <= work.maximum_sequences,
            "too many sequence tables"
        );
        let mut flat = vec![u32::MAX; work.maximum_sequences * work.stride];
        for (sequence, table) in tables.iter().enumerate() {
            ensure!(table.len() <= work.stride, "sequence table exceeds context");
            ensure!(
                table
                    .iter()
                    .all(|&p| (p as usize) < self.capacity / PAGE_TOKENS),
                "physical page outside KV pool"
            );
            flat[sequence * work.stride..sequence * work.stride + table.len()]
                .copy_from_slice(table);
        }
        let mut writes = std::collections::BTreeSet::new();
        let mut ordered = BTreeMap::new();
        let mut maximum = 0;
        work.groups.clear();
        let mut previous_sequence = None;
        for (row_index, row) in rows.iter().enumerate() {
            let position = row.position as usize;
            ensure!((row.token as usize) < vocabulary, "invalid packed token");
            ensure!(position < work.context, "packed position exceeds context");
            let table = tables
                .get(row.sequence as usize)
                .context("missing sequence table")?;
            let page = *table
                .get(position / PAGE_TOKENS)
                .context("missing physical KV page")?;
            ensure!(
                writes.insert(page as usize * PAGE_TOKENS + position % PAGE_TOKENS),
                "duplicate physical KV write"
            );
            if let Some(previous) = ordered.insert(row.sequence, position) {
                ensure!(
                    position == previous + 1,
                    "packed positions must be consecutive within each request"
                );
            }
            maximum = maximum.max(position + 1);
            if previous_sequence == Some(row.sequence) {
                let group = work.groups.last_mut().context("missing query group")?;
                group.count += 1;
                group.maximum = position + 1;
            } else {
                work.groups.push(super::llama_plan::Group {
                    start: row_index,
                    count: 1,
                    maximum: position + 1,
                });
                previous_sequence = Some(row.sequence);
            }
        }
        let decoding = rows.len() <= 8
            && outputs.len() == rows.len()
            && outputs
                .iter()
                .enumerate()
                .all(|(i, &row)| i == row as usize);
        work.span = if decoding {
            super::llama::decode_span(maximum)
        } else {
            256
        };
        work.parts = maximum.div_ceil(work.span);
        if decoding {
            work.parts = super::llama::decode_parts(maximum, work.span);
        }
        // Aligned multi-request prefill uses one direct attention partition.
        // Its plan is independent of the live sequence IDs and positions while
        // this bound holds, and it needs no retained vocabulary output.
        let prefill_graph = !decoding
            && rows.len() > 8
            && outputs.is_empty()
            && work.groups.len() > 1
            && maximum <= 4096
            && self.llama_flash
            && work
                .groups
                .iter()
                .all(|g| g.count > 8 && g.start.is_multiple_of(8));
        let graph_key = if prefill_graph {
            (rows.len(), 1, 4096)
        } else {
            (rows.len(), work.parts, work.span)
        };
        let tokens: Vec<_> = rows.iter().map(|r| r.token).collect();
        let positions: Vec<_> = rows.iter().map(|r| r.position).collect();
        let sequences: Vec<_> = rows.iter().map(|r| r.sequence).collect();
        // Retain every host slice until all queued copies finish, including error paths.
        let queued = (|| -> Result<()> {
            // SAFETY: all five sources remain alive through the unconditional sync below.
            unsafe {
                stage(&self.stream, &work.positions, &positions)?;
                stage(&self.stream, &work.sequences, &sequences)?;
                stage(&self.stream, &work.pages, &flat)?;
                stage(&self.stream, &work.outputs, outputs)?;
                stage(&self.stream, &self.token_ids, &tokens)?;
            }
            Ok(())
        })();
        let synchronized = self.stream.synchronize();
        queued?;
        synchronized.context("finishing paged metadata uploads")?;
        let graph = (decoding || prefill_graph) && self.cuda_graph && self.preallocated;
        if graph {
            if let Some(replay) = self.paged_replays.borrow().get(&graph_key) {
                replay.launch(self)?;
                return receive(self, replay.logits());
            }
        }
        let capture = if graph { self.begin_graph()? } else { None };
        let logits = self.paged_logits(&tokens, outputs)?;
        if let Some(capture) = capture {
            let replay = graph::paged::Replay::new(self, capture, logits)?;
            replay.launch(self)?;
            let result = receive(self, replay.logits());
            let mut cache = self.paged_replays.borrow_mut();
            if cache.len() >= 16 {
                if let Some(&key) = cache.keys().next() {
                    cache.remove(&key);
                }
            }
            cache.insert(graph_key, replay);
            return result;
        }
        receive(self, logits.as_deref())
    }

    fn paged_logits(&self, tokens: &[u32], outputs: &[u32]) -> Result<Option<Buffer>> {
        if tokens.len() > 8
            && self
                .llama_weight_cache
                .keys()
                .any(|name| name.ends_with("attn_qkv.weight"))
        {
            let work = self
                .paging
                .as_ref()
                .context("missing paged rotary workspace")?;
            // SAFETY: configured tables cover the validated packed rows. All
            // layers consume these coefficients before the next batch overwrites them.
            unsafe {
                self.module.llama_position.coefficients(
                    &self.stream,
                    LaunchConfig::for_num_elems((tokens.len() * 64) as u32),
                    tokens.len(),
                    self.rope_base,
                    &work.positions,
                    work.coefficients.cu_deviceptr() as *mut f32,
                )?;
            }
        }
        let selected = if self.llama_fusion && tokens.len() > 8 {
            let Some(selected) = self.llama_fused_hidden(tokens, outputs)? else {
                return Ok(None);
            };
            selected
        } else {
            let mut hidden = self.embed_many(tokens)?;
            for layer in 0..self.geometry.layers {
                let normalized =
                    self.norm(&format!("blk.{layer}.attn_norm.weight"), &hidden, 2048)?;
                let attention = self.llama_attention(layer, &normalized)?;
                drop(normalized);
                let residual = self.add(&hidden, &attention)?;
                drop(hidden);
                drop(attention);
                let normalized = self.norm(
                    &format!("blk.{layer}.post_attention_norm.weight"),
                    &residual,
                    2048,
                )?;
                let down = self.feed_forward(layer, &normalized)?;
                hidden = self.add(&residual, &down)?;
            }
            if outputs.is_empty() {
                return Ok(None);
            }
            self.gather_paged_hidden(&hidden, outputs)?
        };
        let normalized = self.norm("output_norm.weight", &selected, 2048)?;
        Ok(Some(self.mm("output.weight", &normalized)?))
    }

    pub(super) fn gather_paged_hidden(
        &self,
        hidden: &DeviceBuffer<f32>,
        outputs: &[u32],
    ) -> Result<Buffer> {
        let selected = self.zeros(outputs.len() * 2048)?;
        let indices = self.paging.as_ref().context("missing paging workspace")?;
        // SAFETY: validated row indices and disjoint complete 2048-element outputs.
        // The explicit count bounds reads from the reserved index buffer.
        unsafe {
            self.module.llama.llama_gather_hidden(
                &self.stream,
                LaunchConfig::for_num_elems(selected.len() as u32),
                outputs.len(),
                &indices.outputs,
                &hidden,
                selected.cu_deviceptr() as *mut f32,
            )?;
        }
        Ok(selected)
    }
}

unsafe fn stage(stream: &CudaStream, output: &DeviceBuffer<u32>, values: &[u32]) -> Result<()> {
    ensure!(
        values.len() <= output.len(),
        "metadata exceeds reserved staging buffer"
    );
    if values.is_empty() {
        return Ok(());
    }
    // SAFETY: validated extent; caller retains the source until stream synchronization.
    let status = unsafe {
        cuda_core::sys::cuMemcpyHtoDAsync_v2(
            output.cu_deviceptr(),
            values.as_ptr().cast(),
            std::mem::size_of_val(values),
            stream.cu_stream(),
        )
    };
    ensure!(
        status == 0,
        "staging paged metadata: {:?}",
        cuda_core::DriverError(status)
    );
    Ok(())
}
