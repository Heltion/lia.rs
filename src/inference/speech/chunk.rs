//! Retain speech KV across duplex units and exclude the unforwarded lookahead.
use super::*;
use crate::models::minicpm_o45::duplex::drain::Drain;
use crate::sampling::speech::{Policy, Sampler};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    Eos,
    Quiet,
    Limit,
}
pub struct DrainedGeneration {
    pub generation: Generation,
    pub reason: StopReason,
}
impl Engine {
    /// Returned codes have all been consumed by the retained speech KV. The
    /// newest sampled token is excluded, whether it is EOS or normal lookahead.
    pub fn speech_generate_chunk(
        &self,
        cache: &mut Cache,
        conditioning: &[f32],
        sampler: &mut Sampler,
        maximum: usize,
    ) -> Result<Generation> {
        ensure!(
            sampler.parameters().policy == Policy::Chunk,
            "checkpoint speech chunks require the chunk sampling policy"
        );
        Ok(self
            .speech_chunk_inner(cache, conditioning, sampler, maximum, Drain::None)?
            .generation)
    }
    /// Apply the pinned reference's turn-end EOS masking and quiet-run stop.
    /// A limit stop remains distinguishable from a completed EOS/quiet stop.
    pub fn speech_generate_drained(
        &self,
        cache: &mut Cache,
        conditioning: &[f32],
        sampler: &mut Sampler,
        maximum: usize,
        drain: Drain,
    ) -> Result<DrainedGeneration> {
        self.speech_chunk_inner(cache, conditioning, sampler, maximum, drain)
    }
    fn speech_chunk_inner(
        &self,
        cache: &mut Cache,
        conditioning: &[f32],
        sampler: &mut Sampler,
        maximum: usize,
        drain: Drain,
    ) -> Result<DrainedGeneration> {
        ensure!(
            matches!(sampler.parameters().policy, Policy::Chunk | Policy::Vllm)
                && (1..=2048).contains(&maximum)
                && !conditioning.is_empty()
                && conditioning.len().is_multiple_of(768)
                && conditioning.iter().all(|v| v.is_finite()),
            "invalid duplex speech chunk parameters"
        );
        let rows = conditioning.len() / 768;
        ensure!(
            cache.position <= cache.capacity
                && rows + maximum - 1 <= cache.capacity - cache.position,
            "duplex speech chunk exceeds retained cache capacity"
        );
        let prefill = std::time::Instant::now();
        let mut logits = self.speech_logits(cache, conditioning)?;
        self.synchronize()?;
        let prefill_seconds = prefill.elapsed().as_secs_f64();
        let decode = std::time::Instant::now();
        let mut codes = Vec::with_capacity(maximum - 1);
        let mut sampled_tokens = 0;
        let mut finished = false;
        let mut reason = StopReason::Limit;
        for step in 0..maximum {
            let stopping = drain.decision(step, &codes);
            if stopping.stop {
                finished = true;
                reason = StopReason::Quiet;
                break;
            }
            if stopping.mask_eos {
                logits[6561] = f32::NEG_INFINITY;
            }
            let token = sampler.sample(&logits, &codes)?;
            sampled_tokens += 1;
            if token == 6561 {
                finished = true;
                reason = StopReason::Eos;
                break;
            }
            if step + 1 == maximum {
                break;
            }
            logits = self.speech_code_logits(cache, token)?;
            codes.push(token);
        }
        self.synchronize()?;
        Ok(DrainedGeneration {
            generation: Generation {
                codes,
                finished,
                sampled_tokens,
                conditioning_tokens: rows,
                prefill_seconds,
                decode_seconds: decode.elapsed().as_secs_f64(),
            },
            reason,
        })
    }
}
