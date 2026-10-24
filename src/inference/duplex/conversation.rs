//! A persistent native language/audio/speech session on one engine worker.
use super::*;
use crate::{
    models::minicpm_o45::{
        PreparedVoice,
        audio::stream::Streaming,
        duplex::{
            CHUNK_EOS,
            audio::{Audio, Schedule},
            drain::TextTail,
        },
    },
    sampling::{duplex::Sampler, speech::Sampler as SpeechSampler},
    tokenizer::Tokenizer,
};
mod input;
mod render;
pub struct Conversation {
    control: Control,
    sampler: Sampler,
    speech_sampler: SpeechSampler,
    speech: speech::Cache,
    audio: AudioCache,
    processor: Streaming,
    voice: PreparedVoice,
    acoustic: Option<CodecVoiceStream>,
    schedule: Schedule,
    tail: TextTail,
    seed: u64,
    acoustic_chunk: u64,
    turn_units: usize,
    valid: bool,
    max_speak_chars: usize,
}
pub struct Output {
    pub language: Unit,
    pub text: String,
    pub audio: Audio,
    pub speech_codes: usize,
    pub speech_sampled_tokens: usize,
    pub speech_stop: Option<speech::StopReason>,
    pub timings: Timings,
}
#[derive(Default)]
pub struct Timings {
    pub input_seconds: f64,
    pub language_seconds: f64,
    pub speech_prefill_seconds: f64,
    pub speech_decode_seconds: f64,
    pub acoustic_seconds: f64,
}
impl Conversation {
    /// The caller owns model/component loading and the tokenizer-defined system
    /// prefix. Construction resets language KV; this session then owns its use.
    pub fn new(
        engine: &mut Engine,
        prefix: &[u32],
        control: Control,
        sampler: Sampler,
        speech_sampler: SpeechSampler,
        voice: PreparedVoice,
        seed: u64,
    ) -> Result<Self> {
        ensure!(
            !control.active() && !prefix.is_empty(),
            "invalid duplex session prefix/state"
        );
        ensure!(
            speech_sampler.parameters().policy == crate::sampling::speech::Policy::Vllm,
            "native duplex sessions require the reference speech policy"
        );
        let speech = engine.new_speech_cache(4096)?;
        let audio = engine.new_audio_cache()?;
        voice.voice()?;
        let processor = Streaming::new(1000, 1035, 20, true)?;
        engine.reset()?;
        engine
            .forward_many(prefix, false)
            .context("prefilling duplex system prefix")?;
        Ok(Self {
            control,
            sampler,
            speech_sampler,
            speech,
            audio,
            processor,
            voice,
            acoustic: None,
            schedule: Schedule::default(),
            tail: TextTail::default(),
            seed,
            acoustic_chunk: 1,
            turn_units: 0,
            valid: true,
            max_speak_chars: 28,
        })
    }
    pub fn valid(&self) -> bool {
        self.valid
    }
    /// Cancellation invalidates this cache lifetime. Start a new session to
    /// reset all language, encoder, speech and acoustic state together.
    pub fn cancel(&mut self) {
        self.valid = false;
    }
    /// Interrupt at a completed unit boundary, retaining language and input
    /// encoder history while dropping the interrupted output turn's caches.
    pub fn interrupt(&mut self) -> Result<()> {
        ensure!(
            self.valid,
            "cannot interrupt an invalid duplex conversation"
        );
        self.control.finish_turn()?;
        self.speech.reset();
        self.acoustic = None;
        self.schedule.reset();
        self.tail.reset();
        self.turn_units = 0;
        Ok(())
    }
    pub fn requested_samples(&self) -> usize {
        self.processor.requested_samples()
    }
    /// Zero disables the reference's optional text-length chunk cutoff.
    pub fn set_max_speak_chars(&mut self, maximum: usize) {
        self.max_speak_chars = maximum;
    }
    /// Shared entry for audio-only and assembled native audio/video embeddings.
    pub fn embedding_unit(
        &mut self,
        engine: &mut Engine,
        tokenizer: &Tokenizer,
        input: &[f32],
        cancelled: impl Fn() -> bool,
    ) -> Result<Output> {
        ensure!(
            self.valid && !cancelled(),
            "duplex session cancelled or invalid"
        );
        self.valid = false;
        let started = std::time::Instant::now();
        let mut history = self.sampler.history().to_vec();
        let mut turn_ended = self.control.turn_ended();
        let mut chunk_tokens = Vec::new();
        let max_chars = self.max_speak_chars;
        let sampler = &mut self.sampler;
        let mut language = engine.duplex_unit(&mut self.control, input, |logits| {
            ensure!(
                !cancelled(),
                "duplex session cancelled during language generation"
            );
            let mut token = sampler.sample(logits)?;
            if max_chars > 0 && !tokenizer.is_control(token) {
                let mut candidate = chunk_tokens.clone();
                candidate.push(token);
                let text = tokenizer.decode_text(&candidate)?;
                if String::from_utf8_lossy(&text).chars().count() >= max_chars {
                    token = CHUNK_EOS;
                }
            }
            let fed = if token == crate::models::minicpm_o45::duplex::LISTEN && !turn_ended {
                crate::models::minicpm_o45::duplex::TTS_BOS
            } else {
                token
            };
            // Apply normalization before the next draw in this same unit.
            // A cutoff candidate was sampled but is not fed or penalized.
            if token != CHUNK_EOS {
                history.push(fed);
                let drop = history.len().saturating_sub(512);
                history.drain(..drop);
            }
            sampler.set_history(history.clone())?;
            if ![
                crate::models::minicpm_o45::duplex::LISTEN,
                CHUNK_EOS,
                crate::models::minicpm_o45::duplex::CHUNK_TTS_EOS,
            ]
            .contains(&fed)
            {
                turn_ended = fed == crate::models::minicpm_o45::duplex::TURN_EOS;
            }
            if [
                crate::models::minicpm_o45::duplex::LISTEN,
                CHUNK_EOS,
                crate::models::minicpm_o45::duplex::CHUNK_TTS_EOS,
                crate::models::minicpm_o45::duplex::TURN_EOS,
            ]
            .contains(&fed)
            {
                chunk_tokens.clear();
            } else {
                chunk_tokens.push(fed);
            }
            Ok(token)
        })?;
        engine.synchronize()?;
        let language_seconds = started.elapsed().as_secs_f64();
        language.prepare_speech_handoff(&mut self.control)?;
        let text = String::from_utf8_lossy(&tokenizer.decode_text(&language.condition.tokens)?)
            .into_owned();
        self.tail.push(&text);
        let mut rendered = self.render(engine, &language, &cancelled)?;
        rendered.timings.language_seconds = language_seconds;
        self.valid = true;
        Ok(Output {
            language,
            text,
            audio: rendered.audio,
            speech_codes: rendered.codes,
            speech_sampled_tokens: rendered.sampled_tokens,
            speech_stop: rendered.stop,
            timings: rendered.timings,
        })
    }
}
