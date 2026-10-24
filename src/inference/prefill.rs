//! Prefill batching respects both the context workspace and device headroom.
use super::*;

const RESERVE: usize = 576 * 1024 * 1024;
const BYTES_PER_TOKEN: usize = 1024 * 1024;

fn plan(
    remaining: usize,
    maximum: usize,
    position: usize,
    available: usize,
    workspace: usize,
) -> Result<usize> {
    let mut count = remaining.min(maximum);
    while count > 0 {
        let partials = count * 24 * (position + count).div_ceil(256) * 258 * 4;
        let required = count * BYTES_PER_TOKEN + partials.saturating_sub(workspace);
        if required <= available && count * (position + count) <= 8 * 1024 * 1024 {
            return Ok(count);
        }
        if count == 1 {
            break;
        }
        count = if count > 256 {
            (count / 2).div_ceil(256) * 256
        } else {
            count / 2
        };
    }
    Err(anyhow!(
        "insufficient CUDA headroom for prefill after reserving decode buffers"
    ))
}

impl Engine {
    pub fn fp8_weight_workspace_bytes(&self) -> usize {
        self.fp8_work
            .borrow()
            .as_ref()
            .and_then(|w| w.weight_tiles.as_ref())
            .map_or(0, |b| b.len() * 4)
    }

    pub fn prefill_batch(&self) -> usize {
        self.prefill_batch
    }

    pub fn prefill_chunk(&self, remaining: usize) -> Result<usize> {
        if (self.geometry.llama || self.geometry.qwen3) && self.preallocated {
            anyhow::ensure!(remaining > 0, "empty prefill chunk request");
            return Ok(remaining.min(self.prefill_batch));
        }
        if self.preallocated {
            let workspace = self
                .attention_work
                .borrow()
                .as_ref()
                .map_or(0, |b| b.len() * 4);
            return attention_plan::chunk(
                remaining,
                self.prefill_batch,
                self.position,
                workspace,
                self.geometry.query_heads,
                self.attention_mma_decode,
                self.attention_query_tiles,
            )
            .ok_or_else(|| {
                anyhow!("reserved attention workspace cannot accommodate the next prefill token")
            });
        }
        let (mut free, mut total) = (0, 0);
        // SAFETY: the inference thread owns the current CUDA context and both
        // output pointers refer to live, correctly sized host integers.
        let status = unsafe { cuda_core::sys::cuMemGetInfo_v2(&mut free, &mut total) };
        if status != 0 {
            return Err(anyhow!(
                "querying CUDA memory: {:?}",
                cuda_core::DriverError(status)
            ));
        }
        let pooled: usize = self
            .pool
            .borrow()
            .iter()
            .map(|(&n, buffers)| n * 4 * buffers.len())
            .sum();
        let workspace = self
            .attention_work
            .borrow()
            .as_ref()
            .map_or(0, |buffer| buffer.len() * 4);
        let reserve = if self.geometry.mtp {
            RESERVE
        } else {
            128 * 1024 * 1024
        };
        plan(
            remaining,
            self.prefill_batch,
            self.position,
            free.saturating_add(pooled).saturating_sub(reserve),
            workspace,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_planning_preserves_headroom_and_progress() -> Result<()> {
        for position in [0, 1994, 8192, 16000, 32000] {
            for remaining in [1, 143, 1253, 1668, 2048] {
                let available = 1200 * 1024 * 1024;
                let count = plan(remaining, 2048, position, available, 0)?;
                assert!(count > 0 && count <= remaining && count <= 2048);
                let required = count * BYTES_PER_TOKEN
                    + count * 24 * (position + count).div_ceil(256) * 258 * 4;
                assert!(required <= available);
            }
        }
        assert!(plan(1, 2048, 0, 0, 0).is_err());
        assert!(plan(1253, 2048, 1994, 1200 * 1024 * 1024, 0)? < 1253);
        Ok(())
    }
}
