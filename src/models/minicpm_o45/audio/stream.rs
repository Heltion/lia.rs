//! Stateful centered-STFT emission for native duplex input at 16 kHz.
use super::{Features, Normalization, Processor};
use anyhow::{Result, ensure};

#[derive(Clone)]
pub struct Streaming {
    processor: Processor,
    buffer: Vec<f32>,
    chunk_frames: usize,
    first_samples: usize,
    redundancy: usize,
    sliding: bool,
    state: State,
}
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct State {
    pub chunks: usize,
    pub emitted_frames: usize,
    pub processed_samples: usize,
    pub base_frame: usize,
    pub dropped_samples: usize,
}
pub struct Chunk {
    pub features: Features,
    pub stable_frames: usize,
    pub full_frames: usize,
    pub core_start: usize,
    pub core_end: usize,
    pub buffer_samples: usize,
}
impl Streaming {
    pub fn state(&self) -> &State {
        &self.state
    }
    pub fn new(
        chunk_ms: usize,
        first_chunk_ms: usize,
        redundancy_ms: usize,
        sliding: bool,
    ) -> Result<Self> {
        ensure!(
            (10..=30_000).contains(&chunk_ms)
                && chunk_ms.is_multiple_of(10)
                && (25..=30_000).contains(&first_chunk_ms)
                && redundancy_ms <= 1000,
            "invalid streaming mel configuration"
        );
        Ok(Self {
            processor: Processor::default(),
            buffer: Vec::new(),
            chunk_frames: chunk_ms / 10,
            first_samples: (first_chunk_ms * 16 / 160).max(1) * 160,
            redundancy: redundancy_ms / 10,
            sliding,
            state: State::default(),
        })
    }
    /// First-chunk size is rounded down to a complete 160-sample hop.
    pub fn requested_samples(&self) -> usize {
        if self.state.chunks == 0 {
            self.first_samples
        } else {
            self.chunk_frames * 160
        }
    }
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.state = State::default();
    }
    pub fn buffer_samples(&self) -> usize {
        self.buffer.len()
    }
    /// Recompute the bounded window, then emit only stable core plus CNN context.
    /// Cloning this object captures an independent restorable session snapshot.
    pub fn push(&mut self, samples: &[f32], last: bool) -> Result<Chunk> {
        ensure!(
            samples.len() <= 480_000 && samples.iter().all(|v| v.is_finite()),
            "invalid streaming PCM"
        );
        let mut buffer = self.buffer.clone();
        buffer.extend_from_slice(samples);
        let mut state = self.state.clone();
        if self.sliding && buffer.len() >= 480_000 {
            // Match the official trigger, stride and hop-aligned guard policy.
            let guard = 16_000.min(buffer.len() - 160_000);
            let drop = 160_000.min(buffer.len() - guard) / 160 * 160;
            buffer.drain(..drop);
            state.dropped_samples += drop;
            state.base_frame += drop / 160;
        }
        ensure!(
            (400..=480_000).contains(&buffer.len()),
            "streaming mel window must contain 400..=480000 samples"
        );
        let full = self.extract(&buffer)?;
        let stable = full.frames.min((buffer.len() - 200) / 160 + 1);
        let core_start = state.emitted_frames;
        let core_end = core_start
            .checked_add(self.chunk_frames)
            .ok_or_else(|| anyhow::anyhow!("streaming frame counter overflow"))?;
        let features = if stable + state.base_frame >= core_end + self.redundancy || last {
            let start = core_start
                .saturating_sub(self.redundancy)
                .saturating_sub(state.base_frame)
                .min(full.frames);
            let end = (core_end + self.redundancy)
                .saturating_sub(state.base_frame)
                .clamp(start, full.frames);
            state.emitted_frames = core_end;
            Self::slice(&full, start, end)
        } else {
            Self::slice(&full, 0, 0)
        };
        state.processed_samples += samples.len();
        state.chunks += 1;
        self.buffer = buffer;
        self.state = state;
        Ok(Chunk {
            features,
            stable_frames: stable,
            full_frames: full.frames,
            core_start,
            core_end,
            buffer_samples: self.buffer.len(),
        })
    }
    pub fn flush(&mut self) -> Result<Features> {
        if self.buffer.is_empty() {
            return Ok(Features {
                frames: 0,
                values: Vec::new(),
            });
        }
        let full = self.extract(&self.buffer)?;
        let total = self.state.base_frame + full.frames;
        let start = self
            .state
            .emitted_frames
            .saturating_sub(self.state.base_frame)
            .min(full.frames);
        if self.state.emitted_frames < total {
            self.state.emitted_frames = total;
        }
        Ok(Self::slice(&full, start, full.frames))
    }
    fn extract(&self, buffer: &[f32]) -> Result<Features> {
        self.processor.extract(
            buffer,
            if buffer.len() < 80_000 {
                Normalization::Fixed
            } else {
                Normalization::Dynamic
            },
        )
    }
    fn slice(full: &Features, start: usize, end: usize) -> Features {
        let mut values = Vec::with_capacity(80 * (end - start));
        for mel in 0..80 {
            values.extend_from_slice(
                &full.values[mel * full.frames + start..mel * full.frames + end],
            );
        }
        Features {
            frames: end - start,
            values,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_input_preserves_snapshot_and_reset_restores_first_chunk() -> Result<()> {
        let mut stream = Streaming::new(1000, 1035, 20, true)?;
        assert_eq!(stream.requested_samples(), 16480);
        assert!(stream.push(&[0.; 399], false).is_err());
        assert_eq!(stream.state, State::default());
        assert!(stream.push(&[f32::NAN; 400], false).is_err());
        assert_eq!(stream.buffer_samples(), 0);
        assert_eq!(stream.push(&[0.; 400], false)?.features.frames, 0);
        assert_eq!(stream.requested_samples(), 16000);
        let snapshot = stream.clone();
        assert_eq!(stream.flush()?.frames, 2);
        assert_eq!(stream.flush()?.frames, 0);
        assert_eq!(snapshot.state.emitted_frames, 0);
        stream.reset();
        assert_eq!(stream.requested_samples(), 16480);
        assert_eq!(stream.state, State::default());
        Ok(())
    }
}
