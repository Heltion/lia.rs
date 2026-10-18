//! Reserve the GPU working set before readiness, then lease bounded logical buffers.
use super::*;
use anyhow::Context;

const MIB: usize = 1024 * 1024;
const RESERVE: usize = 768 * MIB;
const SMALL_ELEMENTS: usize = 48 * 17408 * 8 + 2 * 248320 * 8;

fn reservation(batch: usize, attention: usize) -> (usize, usize, usize) {
    let scratch = (2 * 17408 + 4 * 6144 + 3 * 5120 + 4 * 1024) * batch * 4 + SMALL_ELEMENTS * 4;
    let packed = batch * 17408 * 2;
    (scratch, attention, packed)
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
            let parts = 248320usize.div_ceil(1024);
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
                self.attention_mma_decode,
                self.attention_query_tiles,
            )
        };
        let maximum = self.prefill_batch.min(self.capacity);
        let batch = [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 1]
            .into_iter()
            .filter(|&n| n <= maximum)
            .find(|&n| {
                let (scratch, attention, packed) = reservation(n, attention_bytes(n));
                scratch + attention + packed + RESERVE <= free
            })
            .ok_or_else(|| anyhow!("insufficient CUDA headroom to reserve inference buffers"))?;
        let (scratch, attention, packed) = reservation(batch, attention_bytes(batch));
        {
            let mut pool = self.pool.borrow_mut();
            for (elements, count) in [
                (17408 * batch, 2),
                (6144 * batch, 4),
                (5120 * batch, 3),
                (1024 * batch, 4),
                (17408 * 8, 48),
                (248320 * 8, 2),
            ] {
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
            checkpoint.hidden = Some(DeviceBuffer::zeroed(&self.stream, 5120)?);
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
        self.attention_query_tiles
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
        for batch in [64, 512, 1024, 2048, 4096] {
            let (scratch, attention, packed) = reservation(batch, 128 * MIB);
            assert!(attention >= 128 * MIB);
            assert!(scratch + attention + packed > previous);
            assert_eq!(packed, batch * 17408 * 2);
            previous = scratch + attention + packed;
        }
    }
}
