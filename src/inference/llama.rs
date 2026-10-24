//! Standard full-attention Llama execution using the shared matrix operators.
use super::*;
use anyhow::Context;

fn decode_override() -> Option<usize> {
    static VALUE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("LIA_LLAMA_DECODE_SPAN")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| (32..=1024).contains(value) && value.is_multiple_of(32))
    })
}

pub(super) fn decode_span(tokens: usize) -> usize {
    if let Some(span) = decode_override() {
        return span;
    }
    if tokens <= 2048 {
        32
    } else if tokens <= 8192 {
        64
    } else {
        160
    }
}

pub(super) fn decode_parts(tokens: usize, span: usize) -> usize {
    let count = tokens.div_ceil(span);
    if count <= 16 {
        count.next_power_of_two()
    } else {
        count.div_ceil(16) * 16
    }
}

pub(super) fn workspace(batch: usize, capacity: usize) -> usize {
    let scalar = batch.min(64) * 16 * capacity.div_ceil(256).next_power_of_two() * 130;
    let prefill = batch.min(1024) * 16 * capacity.div_ceil(4096) * 130;
    let decode =
        batch.min(64) * 16 * capacity.div_ceil(decode_span(capacity)).next_power_of_two() * 130;
    prefill.max(decode).max(scalar) * 4
}

impl Engine {
    pub fn requested_decode_span() -> Option<usize> {
        decode_override()
    }
    pub fn full_prefill_span_enabled(&self) -> bool {
        self.geometry.llama && self.llama_full_prefill_span
    }
    pub fn llama_attention_backend(&self) -> &'static str {
        if self.llama_flash && self.llama_flash_wide {
            "paged_f16_mma_kv64_prefill_f32_accumulation"
        } else if self.llama_flash {
            "paged_f16_mma_prefill_f32_accumulation"
        } else {
            "paged_f32_scalar"
        }
    }
    pub fn tensor_decode_enabled(&self) -> bool {
        self.geometry.llama && self.llama_flash && self.llama_flash_decode
    }
    pub fn bf16_vector_enabled(&self) -> bool {
        self.geometry.llama && self.bf16_vector
    }
    pub(super) fn llama_attention(&self, layer: usize, x: &DeviceBuffer<f32>) -> Result<Buffer> {
        self.llama_attention_prepared(layer, x, false)
    }
    pub(super) fn llama_attention_prepared(
        &self,
        layer: usize,
        x: &DeviceBuffer<f32>,
        prepared: bool,
    ) -> Result<Buffer> {
        self.llama_attention_selected(layer, x, prepared, None)
    }

    pub(super) fn llama_attention_selected(
        &self,
        layer: usize,
        x: &DeviceBuffer<f32>,
        prepared: bool,
        selected: Option<&[u32]>,
    ) -> Result<Buffer> {
        let batch = x.len() / 2048;
        let prefix = format!("blk.{layer}");
        anyhow::ensure!(
            !prepared || batch > 8,
            "packed attention requires prefill rows"
        );
        let combined = self.llama_qkv_prefill(layer, x, prepared)?;
        let raw = if combined.is_some() {
            None
        } else {
            Some(if prepared {
                [
                    self.mm_prepared(&format!("{prefix}.attn_q.weight"), x, true)?,
                    self.mm_prepared(&format!("{prefix}.attn_k.weight"), x, true)?,
                    self.mm_prepared(&format!("{prefix}.attn_v.weight"), x, true)?,
                ]
            } else {
                self.mm_many(
                    [
                        &format!("{prefix}.attn_q.weight"),
                        &format!("{prefix}.attn_k.weight"),
                        &format!("{prefix}.attn_v.weight"),
                    ],
                    x,
                )?
            })
        };
        let q = self.zeros(batch * 2048)?;
        let k = raw.as_ref().map(|_| self.zeros(batch * 256)).transpose()?;
        let out = self.zeros(batch * 2048)?;
        let State::Attention { k: kc, v: vc } = &self.states[layer] else {
            anyhow::bail!("Llama layer has no attention cache");
        };
        let paged = self.paging.as_ref();
        let single = [super::llama_plan::Group {
            start: 0,
            count: batch,
            maximum: self.position + batch,
        }];
        let groups = paged.map_or(single.as_slice(), |p| p.groups.as_slice());
        let maximum = groups
            .iter()
            .map(|g| g.maximum)
            .max()
            .unwrap_or(self.position + batch);
        let tiles = if let Some(selected) = selected {
            selected
                .iter()
                .map(|&row| {
                    let row = row as usize;
                    let group = groups
                        .iter()
                        .find(|g| row >= g.start && row < g.start + g.count)
                        .context("selected attention row outside query groups")?;
                    let maximum = group.maximum - (group.start + group.count - row - 1);
                    let span = decode_span(maximum);
                    Ok(super::llama_plan::Tile {
                        start: row,
                        rows: 1,
                        parts: decode_parts(maximum, span),
                        span,
                        tensor: self.llama_flash && self.llama_flash_decode,
                        compact: self.llama_flash && self.llama_flash_decode,
                    })
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            super::llama_plan::tiles(
                groups,
                batch,
                maximum,
                self.llama_flash,
                self.llama_flash_decode,
                self.llama_full_prefill_span,
            )
        };
        let positions = paged.map_or(&self.token_ids, |p| &p.positions);
        let sequences = paged.map_or(&self.token_ids, |p| &p.sequences);
        let pages = paged.map_or(&self.token_ids, |p| &p.pages);
        let stride = paged.map_or(0, |p| p.stride);
        let required = tiles
            .iter()
            .map(|t| t.rows * 16 * t.parts * 130)
            .max()
            .unwrap_or(0);
        let mut work = self.attention_work.borrow_mut();
        if work.as_ref().is_none_or(|b| b.len() < required) {
            anyhow::ensure!(
                !self.preallocated,
                "reserved Llama attention workspace exceeded"
            );
            work.take();
            *work = Some(DeviceBuffer::zeroed(&self.stream, required)?);
        }
        let partial = work
            .as_ref()
            .ok_or_else(|| anyhow!("missing attention workspace"))?;
        // SAFETY: the validated Llama geometry has128-dimensional heads and2KV
        // heads. The state capacity bounds all written and attended token rows.
        unsafe {
            if let Some(combined) = &combined {
                let paged = paged.context("combined projections require paging")?;
                self.module.llama_position.rotary_cache(
                    &self.stream,
                    LaunchConfig::for_num_elems((batch * 1024) as u32),
                    batch,
                    stride,
                    positions,
                    sequences,
                    pages,
                    &paged.coefficients,
                    combined,
                    q.cu_deviceptr() as *mut f32,
                    kc.cu_deviceptr() as *mut u16,
                    vc.cu_deviceptr() as *mut u16,
                )?;
            } else {
                let [raw_q, raw_k, v] = raw.as_ref().context("missing separate projections")?;
                let k = k.as_ref().context("missing rotary key workspace")?;
                self.module.llama.llama_rope(
                    &self.stream,
                    LaunchConfig::for_num_elems(raw_q.len() as u32),
                    16,
                    self.position,
                    self.rope_base,
                    paged.is_some(),
                    positions,
                    &raw_q,
                    q.cu_deviceptr() as *mut f32,
                )?;
                self.module.llama.llama_rope(
                    &self.stream,
                    LaunchConfig::for_num_elems(raw_k.len() as u32),
                    2,
                    self.position,
                    self.rope_base,
                    paged.is_some(),
                    positions,
                    &raw_k,
                    k.cu_deviceptr() as *mut f32,
                )?;
                if paged.is_some() {
                    self.module.llama.llama_cache_paged(
                        &self.stream,
                        LaunchConfig::for_num_elems(k.len() as u32),
                        stride,
                        positions,
                        sequences,
                        pages,
                        &k,
                        &v,
                        kc.cu_deviceptr() as *mut u16,
                        vc.cu_deviceptr() as *mut u16,
                    )?;
                } else {
                    self.module.rotary.cache_kv_half(
                        &self.stream,
                        LaunchConfig::for_num_elems(k.len() as u32),
                        256,
                        self.position,
                        &k,
                        &v,
                        kc.cu_deviceptr() as *mut u16,
                        vc.cu_deviceptr() as *mut u16,
                    )?;
                }
            }
            for tile in tiles {
                let config = crate::gpu::llama_attention::Config {
                    start: tile.start,
                    queries: tile.rows,
                    position: self.position,
                    parts: tile.parts,
                    span: tile.span,
                    paged: paged.is_some(),
                    stride,
                    tensor: tile.tensor,
                    compact: tile.compact,
                    wide: tile.tensor && !tile.compact && self.llama_flash_wide,
                    direct: tile.tensor && tile.parts == 1,
                };
                crate::gpu::llama_attention::partial(
                    &self.module,
                    &self.stream,
                    config,
                    positions,
                    sequences,
                    pages,
                    &q,
                    kc,
                    vc,
                    if config.direct { &out } else { partial },
                )?;
                if !config.direct {
                    crate::gpu::llama_attention::merge(
                        &self.module,
                        &self.stream,
                        config,
                        partial,
                        &out,
                    )?;
                }
            }
        }
        if let Some(selected) = selected {
            if selected.is_empty() {
                return Ok(q);
            }
            let output = self.gather_paged_hidden(&out, selected)?;
            self.mm(&format!("{prefix}.attn_output.weight"), &output)
        } else {
            self.mm(&format!("{prefix}.attn_output.weight"), &out)
        }
    }
}
