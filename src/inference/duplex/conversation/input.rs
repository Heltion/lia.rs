//! Input tower execution and native video/audio ordering within a model unit.
use super::*;
use crate::models::minicpm_o45::Prepared;
impl Conversation {
    /// First audio unit consumes16480 padded samples, later units16000 at16kHz.
    pub fn audio_unit(
        &mut self,
        engine: &mut Engine,
        tokenizer: &Tokenizer,
        pcm: &[f32],
        cancelled: impl Fn() -> bool,
    ) -> Result<Output> {
        self.audio_video_unit(engine, tokenizer, pcm, &[], cancelled)
    }
    /// Frame groups precede this unit's one audio block. A video-only unit
    /// leaves Whisper's history and first-audio padding state untouched.
    pub fn audio_video_unit(
        &mut self,
        engine: &mut Engine,
        tokenizer: &Tokenizer,
        pcm: &[f32],
        frames: &[Prepared],
        cancelled: impl Fn() -> bool,
    ) -> Result<Output> {
        ensure!(
            self.valid && !cancelled(),
            "duplex session cancelled or invalid"
        );
        ensure!(
            !pcm.is_empty() || !frames.is_empty(),
            "empty duplex input unit"
        );
        ensure!(
            pcm.is_empty() || pcm.len() == self.requested_samples(),
            "incomplete duplex audio unit"
        );
        ensure!(pcm.iter().all(|v| v.is_finite()), "nonfinite duplex PCM");
        self.valid = false;
        let started = std::time::Instant::now();
        let mut input = if frames.is_empty() {
            Vec::new()
        } else {
            engine.duplex_frame_embeddings(frames, tokenizer)?
        };
        if !pcm.is_empty() {
            let prefix = if self.processor.state().chunks == 0 {
                0
            } else {
                2
            };
            let features = self.processor.push(pcm, false)?.features;
            input.extend(engine.audio_stream_embeddings(&mut self.audio, &features, prefix, 2)?);
        }
        let input_seconds = started.elapsed().as_secs_f64();
        ensure!(
            !cancelled(),
            "duplex session cancelled after input encoding"
        );
        self.valid = true;
        let mut output = self.embedding_unit(engine, tokenizer, &input, cancelled)?;
        output.timings.input_seconds = input_seconds;
        Ok(output)
    }
}
