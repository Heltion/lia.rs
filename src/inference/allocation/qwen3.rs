//! Reserve bounded Qwen3 language scratch before serving or graph capture.
use super::*;

fn layout(batch: usize, vocabulary: usize, partial: usize) -> Vec<(usize, usize)> {
    vec![
        (24576 * batch, 2),
        (12288 * batch, 3),
        (6144 * batch, 2),
        (4096 * batch, 8),
        (1024 * batch, 6),
        (vocabulary * batch, 1),
        (vocabulary * 8, 2),
        (24576 * 8, 4),
        (partial, 2),
        // Encoder validation and direct audio embedding calls do not require a
        // language context as long as the 1500-row Whisper feature sequence.
        (1500 * 1024, 8),
        (1500 * 4096, 2),
    ]
}

impl Engine {
    pub(in crate::inference) fn prepare_qwen3_buffers(&mut self) -> Result<()> {
        if !self.preallocated {
            return Ok(());
        }
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: this inference thread owns the bound CUDA context.
        let status = unsafe { cuda_core::sys::cuMemGetInfo_v2(&mut free, &mut total) };
        anyhow::ensure!(
            status == 0,
            "querying Qwen3 reservation headroom: {:?}",
            cuda_core::DriverError(status)
        );
        let partial = 32usize
            .checked_mul(self.capacity.div_ceil(self.attention_decode_span))
            .and_then(|v| v.checked_mul(130))
            .context("Qwen3 attention reservation overflow")?;
        let vocabulary = self.vocabulary_size();
        let maximum = self.prefill_batch.min(self.capacity);
        // The shared component matrix path also serves the 1500-row Whisper
        // encoder. Reserve its bounded input plane even for small text batches.
        let packed_words = |batch: usize| (12288 * batch / 2).max(1500 * 4096 / 2);
        let bytes = |batch| {
            layout(batch, vocabulary, partial)
                .into_iter()
                .map(|(n, count)| n * count * 4)
                .sum::<usize>()
                + packed_words(batch) * 4
        };
        let batch = [
            4096, 3072, 2048, 1536, 1024, 768, 512, 384, 256, 192, 128, 64, 32, 16, 8, 4, 1,
        ]
        .into_iter()
        .filter(|&b| b <= maximum)
        .find(|&b| bytes(b) + 256 * MIB <= free)
        .context("insufficient CUDA headroom for Qwen3 scratch reservation")?;
        {
            let mut pool = self.pool.borrow_mut();
            for (elements, count) in layout(batch, vocabulary, partial) {
                let buffers = pool.entry(elements).or_default();
                for _ in 0..count {
                    buffers.push(DeviceBuffer::zeroed(&self.stream, elements).with_context(
                        || format!("reserving Qwen3 scratch of {elements} elements"),
                    )?);
                }
            }
        }
        *self.prefill_input.borrow_mut() =
            Some(DeviceBuffer::zeroed(&self.stream, packed_words(batch))?);
        self.prefill_batch = batch;
        self.scratch_reserved = bytes(batch);
        if self.cuda_graph && self.qwen3_replay_enabled {
            let mut pool = self.decode_pool.borrow_mut();
            for (elements, count) in layout(1, vocabulary, partial)
                .into_iter()
                .take(5)
                .chain([(24576 * 8, 4), (partial, 2)])
            {
                for _ in 0..count {
                    pool.entry(elements).or_default().push(
                        DeviceBuffer::zeroed(&self.stream, elements)
                            .context("reserving isolated Qwen3 decode scratch")?,
                    );
                    self.scratch_reserved += elements * 4;
                }
            }
        }
        self.stream.synchronize()?;
        eprintln!(
            "Qwen3 buffers reserved: prefill_max={batch} scratch_bytes={}",
            self.scratch_reserved
        );
        Ok(())
    }
}
