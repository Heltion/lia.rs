//! Hidden/text merge and the two terminators used by nonstreaming speech.
use super::*;

impl Engine {
    /// Closed-sentence duplex handoffs terminate the text before audio BOS.
    pub fn duplex_turn_conditioning(
        &self,
        tokens: &[u32],
        hidden: &[f32],
        closed_sentence: bool,
    ) -> Result<Vec<f32>> {
        if !closed_sentence {
            return self.duplex_speech_conditioning(tokens, hidden);
        }
        if !tokens.is_empty() {
            return self.speech_conditioning(tokens, hidden);
        }
        ensure!(
            hidden.is_empty(),
            "empty duplex text has nonempty hidden rows"
        );
        self.speech_embedding("speech.text_embd", &[151692, 151687])?
            .to_host_vec(&self.stream)
            .map_err(Into::into)
    }
    /// Duplex units append only audio BOS; text EOS belongs to whole utterances.
    pub fn duplex_speech_conditioning(&self, tokens: &[u32], hidden: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            hidden.len() == tokens.len() * 4096 && hidden.iter().all(|v| v.is_finite()),
            "invalid duplex speech conditioning rows"
        );
        let bos = self.speech_embedding("speech.text_embd", &[151687])?;
        if tokens.is_empty() {
            return bos.to_host_vec(&self.stream).map_err(Into::into);
        }
        let hidden = DeviceBuffer::from_host(&self.stream, hidden)?;
        let merged = self.speech_merge(tokens, &hidden)?;
        let mut output = self.zeros(merged.len() + 768)?;
        // SAFETY: disjoint complete merged rows and one audio-BOS row fill the output.
        unsafe {
            self.module.basic.concat(
                &self.stream,
                LaunchConfig::for_num_elems(output.len() as u32),
                &merged,
                &bos,
                &mut output,
            )?;
        }
        output.to_host_vec(&self.stream).map_err(Into::into)
    }
    pub fn speech_conditioning(&self, tokens: &[u32], hidden: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            !tokens.is_empty()
                && hidden.len() == tokens.len() * 4096
                && hidden.iter().all(|v| v.is_finite()),
            "invalid speech conditioning extent"
        );
        let input = DeviceBuffer::from_host(&self.stream, hidden)?;
        let merged = self.speech_merge(tokens, &input)?;
        let terminators = self.speech_embedding("speech.text_embd", &[151692, 151687])?;
        let mut output = self.zeros(merged.len() + terminators.len())?;
        // SAFETY: the unique destination covers both complete 768-dimensional spans.
        unsafe {
            self.module.basic.concat(
                &self.stream,
                LaunchConfig::for_num_elems(output.len() as u32),
                &merged,
                &terminators,
                &mut output,
            )?;
        }
        output.to_host_vec(&self.stream).map_err(Into::into)
    }

    pub(super) fn speech_embedding(&self, name: &str, tokens: &[u32]) -> Result<Buffer> {
        let weight = self.weight(name)?;
        ensure!(
            weight.kind == 30
                && weight.shape.len() == 2
                && weight.shape[0] == 768
                && !tokens.is_empty()
                && tokens.len() <= crate::gpu::MAX_PREFILL_BATCH
                && tokens
                    .iter()
                    .all(|&token| (token as usize) < weight.shape[1]),
            "invalid speech embedding indices"
        );
        let ids = DeviceBuffer::from_host(&self.stream, tokens)?;
        let mut output = self.zeros(tokens.len() * 768)?;
        // SAFETY: each validated ID indexes one complete BF16 embedding row.
        unsafe {
            self.module.basic.embedding_many(
                &self.stream,
                LaunchConfig::for_num_elems(output.len() as u32),
                30,
                2,
                1,
                768,
                &ids,
                &weight.data,
                &mut output,
            )?;
        }
        Ok(output)
    }

    fn speech_merge(&self, tokens: &[u32], input: &DeviceBuffer<f32>) -> Result<Buffer> {
        let first = self.component_mm("speech.semantic.1.weight", input, tokens.len(), true)?;
        // SAFETY: consumed output lease and loaded 768-channel projector bias.
        unsafe {
            self.module.whisper.whisper_relu(
                &self.stream,
                LaunchConfig::for_num_elems(first.len() as u32),
                first.len(),
                768,
                self.floats("speech.semantic.1.bias")?,
                first.cu_deviceptr() as *mut f32,
            )?;
        }
        let second = self.component_mm("speech.semantic.2.weight", &first, tokens.len(), true)?;
        let unit = self.zeros(second.len())?;
        // SAFETY: unique affine output and separate complete normalized output rows.
        unsafe {
            self.module.vision_math.affine_in_place::<false>(
                &self.stream,
                LaunchConfig::for_num_elems(second.len() as u32),
                second.len(),
                768,
                self.floats("speech.semantic.2.bias")?,
                second.cu_deviceptr() as *mut f32,
            )?;
            self.module.speech.speech_unit_norm(
                &self.stream,
                LaunchConfig::for_num_elems((tokens.len() * 32) as u32),
                &second,
                unit.cu_deviceptr() as *mut f32,
            )?;
        }
        let text = self.speech_embedding("speech.text_embd", tokens)?;
        self.add(&text, &unit)
    }
}
