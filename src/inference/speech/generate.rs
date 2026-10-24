//! Autoregressive speech-code generation with its own cache and sampler.
use super::*;
use crate::sampling::speech::{Parameters, Sampler};
pub struct Generation {
    pub codes: Vec<u32>,
    pub finished: bool,
    /// Includes a sampled EOS when present; codes excludes that control symbol.
    pub sampled_tokens: usize,
    pub conditioning_tokens: usize,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
}
impl Engine {
    pub fn speech_generate(
        &self,
        conditioning: &[f32],
        parameters: Parameters,
        maximum: usize,
        seed: u64,
    ) -> Result<Generation> {
        self.speech_generate_with(conditioning, parameters, maximum, seed, |_| Ok(()))
    }
    /// Deliver each code as it is confirmed, without buffering it until EOS.
    pub fn speech_generate_with(
        &self,
        conditioning: &[f32],
        parameters: Parameters,
        maximum: usize,
        seed: u64,
        mut emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<Generation> {
        ensure!(
            !conditioning.is_empty()
                && conditioning.len().is_multiple_of(768)
                && (1..=2048).contains(&maximum)
                && parameters.min_tokens <= maximum,
            "invalid speech generation extent"
        );
        let rows = conditioning.len() / 768;
        ensure!(
            rows + maximum <= 4096,
            "speech generation exceeds decoder context"
        );
        let mut sampler = Sampler::new(parameters, seed)?;
        let mut cache = self.new_speech_cache(rows + maximum)?;
        let prefill = std::time::Instant::now();
        let mut logits = self.speech_logits(&mut cache, conditioning)?;
        self.synchronize()?;
        let prefill_seconds = prefill.elapsed().as_secs_f64();
        let decode = std::time::Instant::now();
        let mut codes = Vec::with_capacity(maximum);
        let mut finished = false;
        let mut sampled_tokens = 0;
        let mut callback_seconds = 0.;
        while sampled_tokens < maximum {
            let token = sampler.sample(&logits, &codes)?;
            sampled_tokens += 1;
            if token == 6561 {
                finished = true;
                break;
            }
            let callback = std::time::Instant::now();
            emit(token)?;
            callback_seconds += callback.elapsed().as_secs_f64();
            codes.push(token);
            if sampled_tokens < maximum {
                logits = self.speech_code_logits(&mut cache, token)?;
            }
        }
        self.synchronize()?;
        Ok(Generation {
            codes,
            finished,
            sampled_tokens,
            conditioning_tokens: rows,
            prefill_seconds,
            decode_seconds: (decode.elapsed().as_secs_f64() - callback_seconds).max(0.),
        })
    }
}
