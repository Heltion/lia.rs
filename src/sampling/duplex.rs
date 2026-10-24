//! Checkpoint duplex sampling: independent chunk-EOS draw, then filtered text.
use super::random::Random;
use anyhow::{Result, ensure};
mod distribution;
use distribution::{argmax, categorical, filter, probabilities};
const VOCABULARY: usize = 151748;
const CHUNK_EOS: u32 = 151718;
#[derive(Clone, Copy, Debug)]
pub struct Parameters {
    pub greedy: bool,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f64,
    pub repetition_penalty: f32,
    pub repetition_window: usize,
    /// The checkpoint StreamDecoder uses tokenizer EOS here, independently of
    /// the model's duplex LISTEN control marker. Callers must choose explicitly.
    pub listen_id: u32,
    pub listen_scale: f32,
    pub listen_top_k: Option<usize>,
}
impl Default for Parameters {
    fn default() -> Self {
        Self {
            greedy: false,
            temperature: 0.7,
            top_k: 100,
            top_p: 0.8,
            repetition_penalty: 1.05,
            repetition_window: 512,
            listen_id: 151645,
            listen_scale: 1.,
            listen_top_k: None,
        }
    }
}
pub struct Sampler {
    parameters: Parameters,
    forbidden: Vec<u32>,
    special: Vec<u32>,
    history: Vec<u32>,
    random: Random,
}
impl Sampler {
    pub fn new(
        parameters: Parameters,
        mut forbidden: Vec<u32>,
        special: Vec<u32>,
        seed: u64,
    ) -> Result<Self> {
        let p = parameters;
        ensure!(
            p.temperature.is_finite()
                && p.temperature > 0.
                && p.top_k <= VOCABULARY
                && p.top_p.is_finite()
                && p.top_p > 0.
                && p.top_p <= 1.
                && p.repetition_penalty.is_finite()
                && p.repetition_penalty > 0.
                && (p.listen_id as usize) < VOCABULARY
                && p.listen_scale.is_finite()
                && p.listen_scale >= 0.
                && forbidden
                    .iter()
                    .chain(&special)
                    .all(|&v| (v as usize) < VOCABULARY),
            "invalid duplex sampling parameters"
        );
        forbidden.push(CHUNK_EOS);
        forbidden.sort_unstable();
        forbidden.dedup();
        Ok(Self {
            parameters,
            forbidden,
            special,
            history: Vec::new(),
            random: Random::new((seed ^ (seed >> 32)) as u32),
        })
    }
    pub fn history(&self) -> &[u32] {
        &self.history
    }
    pub fn set_history(&mut self, history: Vec<u32>) -> Result<()> {
        ensure!(
            history.iter().all(|&v| (v as usize) < VOCABULARY),
            "invalid duplex repetition history"
        );
        self.history = history;
        Ok(())
    }
    pub fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        let random = &mut self.random;
        run(
            self.parameters,
            &self.forbidden,
            &self.special,
            &mut self.history,
            logits,
            |_| Ok(random.uniform()),
        )
    }
    /// Expose each actual categorical distribution at the random-draw boundary
    /// for reference validation without substituting model or policy decisions.
    pub fn sample_with(
        &mut self,
        logits: &[f32],
        draw: impl FnMut(&[f64]) -> Result<f64>,
    ) -> Result<u32> {
        run(
            self.parameters,
            &self.forbidden,
            &self.special,
            &mut self.history,
            logits,
            draw,
        )
    }
}
fn run(
    p: Parameters,
    forbidden: &[u32],
    special: &[u32],
    history: &mut Vec<u32>,
    logits: &[f32],
    mut draw: impl FnMut(&[f64]) -> Result<f64>,
) -> Result<u32> {
    ensure!(
        logits.len() == VOCABULARY && logits.iter().all(|v| !v.is_nan() && *v != f32::INFINITY),
        "invalid duplex logits"
    );
    let initial = if p.greedy {
        argmax(logits)?
    } else {
        categorical(&probabilities(logits)?, &mut draw)?
    };
    if initial == CHUNK_EOS {
        return Ok(initial);
    }
    let mut scores = logits.to_vec();
    for &id in forbidden {
        scores[id as usize] = f32::NEG_INFINITY;
    }
    if p.repetition_penalty != 1. && !history.is_empty() {
        let mut seen = vec![false; VOCABULARY];
        let count = if p.repetition_window == 0 {
            history.len()
        } else {
            p.repetition_window
        };
        for &id in history.iter().rev().take(count) {
            if !seen[id as usize] {
                seen[id as usize] = true;
                let value = &mut scores[id as usize];
                // The released text policy divides negative logits as well.
                if p.repetition_penalty > 1. {
                    *value /= p.repetition_penalty;
                } else {
                    *value *= 1. / p.repetition_penalty;
                }
            }
        }
    }
    if p.listen_scale != 1. {
        scores[p.listen_id as usize] *= p.listen_scale;
    }
    ensure!(
        scores.iter().all(|v| !v.is_nan() && *v != f32::INFINITY),
        "duplex score transformation overflow"
    );
    let rank = scores
        .iter()
        .filter(|&&v| v > scores[p.listen_id as usize])
        .count();
    if p.listen_top_k.is_some_and(|k| rank < k) {
        return Ok(p.listen_id);
    }
    let token = if p.greedy {
        argmax(&scores)?
    } else {
        for v in &mut scores {
            *v /= p.temperature;
        }
        filter(&mut scores, p.top_k, p.top_p)?;
        categorical(&probabilities(&scores)?, &mut draw)?
    };
    if !special.contains(&token) {
        history.push(token);
    }
    Ok(token)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn early_eos_preserves_history_and_negative_penalty_matches_released_policy() -> Result<()> {
        let mut sampler = Sampler::new(
            Parameters {
                greedy: true,
                repetition_penalty: 1.1,
                ..Parameters::default()
            },
            Vec::new(),
            Vec::new(),
            42,
        )?;
        sampler.set_history(vec![1000])?;
        let mut logits = vec![f32::NEG_INFINITY; VOCABULARY];
        logits[CHUNK_EOS as usize] = 0.;
        assert_eq!(sampler.sample(&logits)?, CHUNK_EOS);
        assert_eq!(sampler.history(), [1000]);
        logits[CHUNK_EOS as usize] = f32::NEG_INFINITY;
        logits[1000] = -1.;
        logits[1001] = -0.96;
        assert_eq!(sampler.sample(&logits)?, 1000);
        assert_eq!(sampler.history(), [1000, 1000]);
        assert!(sampler.sample(&[]).is_err());
        assert_eq!(sampler.history(), [1000, 1000]);
        Ok(())
    }
    #[test]
    fn top_k_ties_are_retained_and_invalid_draws_do_not_change_history() -> Result<()> {
        let mut sampler = Sampler::new(
            Parameters {
                top_k: 2,
                top_p: 1.,
                ..Parameters::default()
            },
            Vec::new(),
            Vec::new(),
            42,
        )?;
        let mut logits = vec![f32::NEG_INFINITY; VOCABULARY];
        logits[1000..1003].fill(1.);
        let mut calls = 0;
        sampler.sample_with(&logits, |p| {
            calls += 1;
            assert_eq!(p.iter().filter(|&&v| v > 0.).count(), 3);
            Ok(0.5)
        })?;
        assert_eq!(calls, 2);
        let history = sampler.history().to_vec();
        assert!(sampler.sample_with(&logits, |_| Ok(1.)).is_err());
        assert_eq!(sampler.history(), history);
        Ok(())
    }
}
