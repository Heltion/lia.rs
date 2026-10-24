//! Retained talker and bounded acoustic caches, with turn-end drain and flush.
use super::*;
use crate::models::minicpm_o45::duplex::{
    audio::packet,
    drain::{Drain, budget},
};
pub(super) struct Rendered {
    pub(super) audio: Audio,
    pub(super) codes: usize,
    pub(super) sampled_tokens: usize,
    pub(super) stop: Option<speech::StopReason>,
    pub(super) timings: Timings,
}
impl Conversation {
    pub(super) fn render(
        &mut self,
        engine: &Engine,
        language: &Unit,
        cancelled: &impl Fn() -> bool,
    ) -> Result<Rendered> {
        if language.condition.tokens.is_empty() && !language.end_of_turn {
            return Ok(Rendered {
                audio: packet(&[], false)?,
                codes: 0,
                sampled_tokens: 0,
                stop: None,
                timings: Timings::default(),
            });
        }
        ensure!(
            !cancelled(),
            "duplex session cancelled before speech generation"
        );
        let first = self.turn_units == 0;
        let sentence = language.end_of_turn && self.tail.ends_sentence();
        let conditioning_start = std::time::Instant::now();
        let conditioning = engine.duplex_turn_conditioning(
            &language.condition.tokens,
            &language.condition.hidden,
            sentence,
        )?;
        let conditioning_seconds = conditioning_start.elapsed().as_secs_f64();
        let (maximum, minimum) = budget(first, language.end_of_turn);
        self.speech_sampler.set_min_tokens(minimum)?;
        let generated = engine.speech_generate_drained(
            &mut self.speech,
            &conditioning,
            &mut self.speech_sampler,
            maximum,
            Drain::for_turn(true, language.end_of_turn, sentence),
        )?;
        let codes = &generated.generation.codes;
        let acoustic_start = std::time::Instant::now();
        if self.acoustic.is_none() {
            let voice = self.voice.voice()?;
            let noise = crate::sampling::gaussian(self.seed, voice.mel().len());
            self.acoustic = Some(engine.new_codec_voice_stream(&voice, 10, &noise)?);
        }
        let mut wave = Vec::new();
        for block in self.schedule.push(codes, language.end_of_turn, first)? {
            ensure!(
                !cancelled(),
                "duplex session cancelled during acoustic synthesis"
            );
            let cache = self
                .acoustic
                .as_mut()
                .context("missing duplex acoustic cache")?;
            let frames = (block.codes.len() - if block.last { 0 } else { 3 }) * 2;
            let seed = self.seed.wrapping_add(self.acoustic_chunk);
            let flow = crate::sampling::gaussian(seed, frames * 80);
            let hift = crate::sampling::gaussian(
                seed ^ 0x9e37_79b9_7f4a_7c15,
                (frames + cache.overlap_frames()) * 480 * 9,
            );
            wave.extend(engine.codec_voice_stream_chunk(
                cache,
                &block.codes,
                &flow,
                &hift,
                block.last,
            )?);
            self.acoustic_chunk = self.acoustic_chunk.wrapping_add(1);
        }
        let audio = packet(&wave, language.end_of_turn)?;
        let acoustic_seconds = acoustic_start.elapsed().as_secs_f64();
        self.turn_units += 1;
        if language.end_of_turn {
            self.speech.reset();
            self.acoustic = None;
            self.schedule.reset();
            self.tail.reset();
            self.turn_units = 0;
        }
        Ok(Rendered {
            audio,
            codes: codes.len(),
            sampled_tokens: generated.generation.sampled_tokens,
            stop: Some(generated.reason),
            timings: Timings {
                speech_prefill_seconds: conditioning_seconds + generated.generation.prefill_seconds,
                speech_decode_seconds: generated.generation.decode_seconds,
                acoustic_seconds,
                ..Default::default()
            },
        })
    }
}
