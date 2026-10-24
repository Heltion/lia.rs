//! Optional prefill shadows retain the original BF16 weights for decode.
use super::*;
use anyhow::{Context, ensure};

impl Engine {
    pub fn prefill_weight_cache_bytes(&self) -> usize {
        self.llama_weight_cache
            .values()
            .map(DeviceBuffer::len)
            .sum()
    }

    pub(super) fn prepare_llama_weight_cache(&mut self) -> Result<()> {
        let requested = std::env::var("LIA_LLAMA_WEIGHT_CACHE_MIB")
            .map(|value| {
                value
                    .parse::<usize>()
                    .context("parsing prefill weight cache budget")
            })
            .unwrap_or(Ok(4096))?;
        if !self.geometry.llama || requested == 0 {
            return Ok(());
        }
        let requested = requested
            .checked_mul(1024 * 1024)
            .context("prefill weight cache budget overflow")?;
        let (mut free, mut total) = (0, 0);
        // SAFETY: current owning CUDA context and live output integers.
        let status = unsafe { cuda_core::sys::cuMemGetInfo_v2(&mut free, &mut total) };
        ensure!(
            status == 0,
            "querying prefill weight cache headroom: {:?}",
            cuda_core::DriverError(status)
        );
        let dynamic = if self.preallocated {
            0
        } else {
            let batch = self.prefill_batch.min(self.capacity);
            let (scratch, attention, packed) = allocation::reservation(
                self.geometry,
                batch,
                llama::workspace(batch, self.capacity),
            );
            scratch + attention + packed
        };
        let limit = requested.min(free.saturating_sub(dynamic + 512 * 1024 * 1024));
        let mut groups = Vec::new();
        for layer in 0..self.geometry.layers {
            groups.push(vec![
                format!("blk.{layer}.ffn_gate.weight"),
                format!("blk.{layer}.ffn_up.weight"),
            ]);
        }
        for name in ["ffn_down", "attn_q", "attn_output", "attn_k", "attn_v"] {
            for layer in 0..self.geometry.layers {
                groups.push(vec![format!("blk.{layer}.{name}.weight")]);
            }
        }
        let mut used = 0;
        for names in groups {
            let bytes = names
                .iter()
                .map(|name| self.weights[name].data.len())
                .sum::<usize>();
            if used + bytes > limit {
                continue;
            }
            for name in names {
                let weight = &self.weights[&name];
                ensure!(
                    weight.kind == 30 && weight.data.len().is_multiple_of(16),
                    "invalid BF16 prefill shadow source {name}"
                );
                let packed = DeviceBuffer::zeroed(&self.stream, weight.data.len())
                    .with_context(|| format!("reserving prefill shadow {name}"))?;
                // SAFETY: aligned complete eight-BF16 groups and disjoint equal-sized output.
                unsafe {
                    self.module.llama_bf16.llama_pack_weights(
                        &self.stream,
                        LaunchConfig::for_num_elems((weight.data.len() / 16) as u32),
                        &weight.data,
                        packed.cu_deviceptr() as *mut u8,
                    )?;
                }
                self.llama_weight_cache.insert(name, packed);
            }
            used += bytes;
        }
        self.stream
            .synchronize()
            .context("preparing prefill weight shadows")?;
        self.combine_llama_projections()?;
        self.scratch_reserved += used;
        eprintln!(
            "Prefill weight shadows reserved: bytes={used} limit={limit}; decode weights remain BF16"
        );
        Ok(())
    }

    fn combine_llama_projections(&mut self) -> Result<()> {
        for layer in 0..self.geometry.layers {
            let names = ["q", "k", "v"].map(|name| format!("blk.{layer}.attn_{name}.weight"));
            if !names
                .iter()
                .all(|name| self.llama_weight_cache.contains_key(name))
            {
                continue;
            }
            let [q, k, v] = names.each_ref().map(|name| &self.llama_weight_cache[name]);
            let bytes = q.len() + k.len() + v.len();
            let combined = DeviceBuffer::zeroed(&self.stream, bytes)
                .context("reserving combined prefill projections")?;
            // SAFETY: aligned half matrices are copied as disjoint complete
            // words. Keep every source alive until the copy has completed.
            unsafe {
                self.module.llama_position.concatenate(
                    &self.stream,
                    LaunchConfig::for_num_elems((bytes / 4) as u32),
                    q,
                    k,
                    v,
                    combined.cu_deviceptr() as *mut u32,
                )?;
            }
            self.stream
                .synchronize()
                .context("combining prefill projections")?;
            for name in names {
                self.llama_weight_cache.remove(&name);
            }
            self.llama_weight_cache
                .insert(format!("blk.{layer}.attn_qkv.weight"), combined);
        }
        Ok(())
    }
}
