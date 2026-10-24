//! MiniCPM-o codec sampling, including both pinned upstream filtering policies.
use super::random::Random;
use anyhow::{Result, ensure};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// Checkpoint generate(): nucleus then top-k, minimum three; first frame unfiltered.
    Checkpoint,
    /// Pinned vLLM-Omni: censor, penalty, temperature, top-k, then nucleus.
    Vllm,
    /// Checkpoint generate_chunk(): temperature/penalty, without top-k/top-p.
    Chunk,
}
#[derive(Clone, Copy, Debug)]
pub struct Parameters {
    pub temperature: f32,
    pub top_p: f64,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub min_tokens: usize,
    pub policy: Policy,
}
impl Default for Parameters {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.85,
            top_k: 25,
            repetition_penalty: 1.05,
            min_tokens: 50,
            policy: Policy::Checkpoint,
        }
    }
}
pub struct Sampler {
    parameters: Parameters,
    random: Random,
}
impl Parameters {
    pub fn validate(self) -> Result<Self> {
        ensure!(
            self.temperature.is_finite()
                && self.temperature > 0.0
                && self.top_p.is_finite()
                && self.top_p > 0.0
                && self.top_p <= 1.0
                && self.top_k <= 6562
                && self.repetition_penalty.is_finite()
                && self.repetition_penalty > 0.0
                && self.min_tokens <= 2048,
            "invalid speech sampling parameters"
        );
        Ok(self)
    }
}
impl Sampler {
    pub fn new(parameters: Parameters, seed: u64) -> Result<Self> {
        Ok(Self {
            parameters: parameters.validate()?,
            random: Random::new((seed ^ (seed >> 32)) as u32),
        })
    }
    pub fn parameters(&self) -> Parameters {
        self.parameters
    }
    pub fn set_min_tokens(&mut self, minimum: usize) -> Result<()> {
        ensure!(minimum <= 2048, "invalid speech minimum token count");
        self.parameters.min_tokens = minimum;
        Ok(())
    }
    /// Dense codec probabilities, in token ID order; no random state is advanced.
    pub fn probabilities(&self, logits: &[f32], history: &[u32]) -> Result<Vec<f64>> {
        ensure!(
            logits.len() == 6562
                && logits.iter().all(|v| !v.is_nan() && *v != f32::INFINITY)
                && history.iter().all(|&id| id < 6561),
            "invalid speech logits or history"
        );
        let p = self.parameters;
        let mut scores = logits.to_vec();
        if p.policy == Policy::Vllm && history.len() < p.min_tokens {
            scores[6561] = f32::NEG_INFINITY;
        }
        if p.policy != Policy::Vllm {
            for v in &mut scores {
                *v /= p.temperature;
            }
        }
        let mut counts = [0u8; 6562];
        for &id in history.iter().rev().take(16) {
            counts[id as usize] += 1;
        }
        for (score, count) in scores.iter_mut().zip(counts) {
            if count != 0 {
                let alpha = p.repetition_penalty.powi(i32::from(count));
                *score = if *score < 0.0 {
                    *score * alpha
                } else {
                    *score / alpha
                };
            }
        }
        if p.policy == Policy::Vllm {
            for v in &mut scores {
                *v /= p.temperature;
            }
        }
        ensure!(
            scores.iter().all(|v| !v.is_nan() && *v != f32::INFINITY),
            "speech penalty overflow"
        );
        let mut candidates: Vec<(usize, f32)> = scores
            .into_iter()
            .enumerate()
            .filter(|(_, v)| *v != f32::NEG_INFINITY)
            .collect();
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        ensure!(!candidates.is_empty(), "empty speech distribution");
        match p.policy {
            Policy::Checkpoint if !history.is_empty() => {
                nucleus(&mut candidates, p.top_p, 3);
                top_k(&mut candidates, p.top_k, 3);
            }
            Policy::Vllm => {
                top_k(&mut candidates, p.top_k, 1);
                nucleus(&mut candidates, p.top_p, 1);
            }
            _ => {}
        }
        if p.policy != Policy::Vllm && history.len() < p.min_tokens {
            candidates.retain(|(id, _)| *id != 6561);
        }
        ensure!(
            !candidates.is_empty(),
            "speech censor removed every candidate"
        );
        let maximum = candidates[0].1 as f64;
        let total: f64 = candidates
            .iter()
            .map(|(_, v)| (f64::from(*v) - maximum).exp())
            .sum();
        let mut output = vec![0.0; 6562];
        for (id, value) in candidates {
            output[id] = (f64::from(value) - maximum).exp() / total;
        }
        Ok(output)
    }
    pub fn sample(&mut self, logits: &[f32], history: &[u32]) -> Result<u32> {
        let probabilities = self.probabilities(logits, history)?;
        let sample = self.random.uniform();
        let mut total = 0.0;
        let mut last = 0;
        for (id, p) in probabilities.into_iter().enumerate() {
            if p > 0.0 {
                last = id as u32;
                total += p;
                if sample < total {
                    return Ok(id as u32);
                }
            }
        }
        Ok(last)
    }
}
fn top_k(values: &mut Vec<(usize, f32)>, k: usize, minimum: usize) {
    if k == 0 || values.len() <= k.max(minimum) {
        return;
    }
    let threshold = values[k.max(minimum) - 1].1;
    values.retain(|(_, v)| *v >= threshold);
}
fn nucleus(values: &mut Vec<(usize, f32)>, p: f64, minimum: usize) {
    if p == 1.0 {
        return;
    }
    let maximum = values[0].1 as f64;
    let total: f64 = values
        .iter()
        .map(|(_, v)| (f64::from(*v) - maximum).exp())
        .sum();
    let mut mass = 0.0;
    let mut keep = values.len();
    for (index, (_, v)) in values.iter().enumerate() {
        mass += (f64::from(*v) - maximum).exp();
        if mass >= p * total && index + 1 >= minimum {
            keep = index + 1;
            break;
        }
    }
    values.truncate(keep);
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn first_frame_window_and_filter_order_remain_distinct() -> Result<()> {
        let mut logits = vec![-100.; 6562];
        for (i, value) in logits[..40].iter_mut().enumerate() {
            *value = 4.0 - i as f32 * 0.02;
        }
        let checkpoint = Sampler::new(Parameters::default(), 42)?;
        let reference = Sampler::new(
            Parameters {
                policy: Policy::Vllm,
                ..Parameters::default()
            },
            42,
        )?;
        let full = checkpoint.probabilities(&logits, &[])?;
        let filtered = reference.probabilities(&logits, &[])?;
        assert!(full.iter().filter(|&&p| p > 0.0).count() > 25);
        assert!(filtered.iter().filter(|&&p| p > 0.0).count() <= 25);
        assert_eq!(full[6561], 0.0);
        assert_eq!(
            checkpoint.probabilities(&logits, &[1; 16])?,
            checkpoint
                .probabilities(&logits, &[2].into_iter().chain([1; 16]).collect::<Vec<_>>())?
        );
        let mut ties = vec![-100.; 6562];
        ties[..30].fill(1.0);
        let uniform = Sampler::new(
            Parameters {
                top_p: 1.0,
                top_k: 25,
                policy: Policy::Vllm,
                ..Parameters::default()
            },
            42,
        )?;
        assert_eq!(
            uniform
                .probabilities(&ties, &[])?
                .iter()
                .filter(|&&p| p > 0.)
                .count(),
            30
        );
        Ok(())
    }
}
