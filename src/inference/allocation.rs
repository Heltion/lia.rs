//! Reserve the GPU working set before readiness, then lease bounded logical buffers.
use super::*;
use anyhow::Context;

const MIB: usize = 1024 * 1024;
const RESERVE: usize = 512 * MIB;
const SMALL_ELEMENTS: usize = 48 * 17408 * 8 + 2 * 248320 * 8;

fn scratch_layout(geometry: geometry::Geometry, batch: usize) -> Vec<(usize, usize)> {
    if geometry.moe {
        vec![
            (8 * geometry.hidden * batch, 1),
            (geometry.recurrent_width() * batch, 2),
            (geometry.value_heads * 128 * batch, 3),
            (geometry.hidden * batch, 6),
            (512 * batch, 4),
            (256 * batch, 1),
        ]
    } else {
        vec![
            (17408 * batch, 2),
            (6144 * batch, 4),
            (5120 * batch, 3),
            (1024 * batch, 4),
        ]
    }
}

fn reservation(
    geometry: geometry::Geometry,
    batch: usize,
    attention: usize,
) -> (usize, usize, usize) {
    let scratch = scratch_layout(geometry, batch)
        .into_iter()
        .map(|(elements, count)| elements * count * 4)
        .sum::<usize>()
        + SMALL_ELEMENTS * 4;
    // Dense projections need one FP16 plane. INT4 experts also preserve the
    // FP16 residual of their larger routed activation in the same workspace.
    let packed_words = if geometry.moe {
        geometry.recurrent_width() / 2
    } else {
        17408 / 2
    };
    (scratch, attention, batch * packed_words * 4)
}

impl Engine {
    pub(super) fn prepare_small_workspaces(&mut self) -> Result<()> {
        let widths: std::collections::BTreeSet<usize> = self
            .weights
            .values()
            .filter(|w| w.shape.len() == 2 && w.kind != 0 && w.kind != 30)
            .map(|w| w.shape[0])
            .collect();
        let mut quant = self.quant_work.borrow_mut();
        let mut sampling = self.sampling_work.borrow_mut();
        for batch in 1..=8 {
            for &width in &widths {
                let elements = width * batch;
                if let std::collections::btree_map::Entry::Vacant(entry) = quant.entry(elements) {
                    entry.insert(QuantWork {
                        packed: DeviceBuffer::zeroed(&self.stream, elements / 4)?,
                        scales: DeviceBuffer::zeroed(&self.stream, elements / 32)?,
                    });
                }
            }
            let parts = self.vocabulary_size().div_ceil(1024);
            self.greedy_work
                .insert(batch, DeviceBuffer::zeroed(&self.stream, batch * parts)?);
            sampling.insert(
                batch,
                SelectionWork {
                    candidates: DeviceBuffer::zeroed(&self.stream, batch * parts * 20)?,
                },
            );
        }
        Ok(())
    }

    pub(super) fn prepare_buffers(&mut self) -> Result<()> {
        if !self.preallocated {
            return Ok(());
        }
        let (mut free, mut total) = (0, 0);
        // SAFETY: the inference thread owns the current CUDA context; both output
        // pointers name live host integers of the driver API's size_t type.
        let status = unsafe { cuda_core::sys::cuMemGetInfo_v2(&mut free, &mut total) };
        anyhow::ensure!(
            status == 0,
            "querying CUDA headroom: {:?}",
            cuda_core::DriverError(status)
        );
        let attention_bytes = |batch| {
            attention_plan::reserved_bytes(
                batch,
                self.capacity,
                self.geometry.query_heads,
                self.attention_mma_decode,
                self.attention_query_tiles,
            )
        };
        // Leave space for the server checkpoint and driver allocations after
        // reserving scratch. Draft state itself was allocated during initialization.
        let reserve = if self.geometry.mtp {
            RESERVE
        } else {
            256 * MIB
        };
        let maximum = self.prefill_batch.min(self.capacity);
        let batch = [
            4096, 3072, 2048, 1536, 1024, 768, 512, 384, 256, 192, 128, 64, 32, 16, 8, 4, 1,
        ]
            .into_iter()
            .filter(|&n| n <= maximum)
            .find(|&n| {
                let (scratch, attention, packed) = reservation(self.geometry, n, attention_bytes(n));
                scratch + attention + packed + reserve <= free
            })
            .ok_or_else(|| anyhow!("insufficient CUDA headroom to reserve inference buffers: free={free}, reserve={reserve}"))?;
        let (scratch, attention, packed) =
            reservation(self.geometry, batch, attention_bytes(batch));
        {
            let mut pool = self.pool.borrow_mut();
            let mut layout = scratch_layout(self.geometry, batch);
            layout.extend([(17408 * 8, 48), (self.vocabulary_size() * 8, 2)]);
            for (elements, count) in layout {
                let buffers = pool.entry(elements).or_default();
                for _ in 0..count {
                    buffers.push(
                        DeviceBuffer::zeroed(&self.stream, elements)
                            .with_context(|| format!("reserving {elements} scratch elements"))?,
                    );
                }
            }
        }
        *self.attention_work.borrow_mut() =
            Some(DeviceBuffer::zeroed(&self.stream, attention / 4)?);
        *self.prefill_input.borrow_mut() = Some(DeviceBuffer::zeroed(&self.stream, packed / 4)?);
        self.prefill_batch = batch;
        self.scratch_reserved = scratch;
        self.stream.synchronize()?;
        eprintln!(
            "GPU buffers reserved: prefill_max={batch} scratch_bytes={scratch} attention_bytes={attention} packed_bytes={packed}"
        );
        Ok(())
    }

    pub(super) fn prepare_tokens(&self, tokens: &[u32]) -> Result<()> {
        anyhow::ensure!(
            !tokens.is_empty() && tokens.len() <= self.token_ids.len(),
            "invalid token staging extent"
        );
        // SAFETY: both extents are validated. The source borrow is kept alive
        // until the stream has completed the copy. This happens before capture.
        let status = unsafe {
            cuda_core::sys::cuMemcpyHtoDAsync_v2(
                self.token_ids.cu_deviceptr(),
                tokens.as_ptr().cast(),
                std::mem::size_of_val(tokens),
                self.stream.cu_stream(),
            )
        };
        anyhow::ensure!(
            status == 0,
            "staging token IDs: {:?}",
            cuda_core::DriverError(status)
        );
        self.stream.synchronize()?;
        Ok(())
    }

    pub(crate) fn prepare_checkpoint(&self) -> Result<Snapshot> {
        let mut checkpoint = self.snapshot()?;
        if checkpoint.hidden.is_none() {
            checkpoint.hidden = Some(DeviceBuffer::zeroed(&self.stream, self.geometry.hidden)?);
        }
        Ok(checkpoint)
    }

    pub fn attention_reserved_bytes(&self) -> usize {
        self.attention_work
            .borrow()
            .as_ref()
            .map_or(0, |b| b.len() * 4)
    }

    pub fn query_tiled_attention(&self) -> bool {
        self.attention_query_tiles != 0
    }

    pub fn gdn_parallel(&self) -> bool {
        self.gdn_prepared.is_some()
    }

    pub fn buffers_preallocated(&self) -> bool {
        self.preallocated
    }
    pub fn scratch_reserved_bytes(&self) -> usize {
        self.scratch_reserved
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reservation_sizes_cover_decode_and_grow_with_prefill() {
        let mut previous = 0;
        for batch in [64, 192, 384, 768, 1024, 1536, 2048, 3072, 4096] {
            let (scratch, attention, packed) =
                reservation(geometry::Geometry::dense(), batch, 128 * MIB);
            assert!(attention >= 128 * MIB);
            assert!(scratch + attention + packed > previous);
            assert_eq!(packed, batch * 17408 * 2);
            previous = scratch + attention + packed;
        }
    }
}
