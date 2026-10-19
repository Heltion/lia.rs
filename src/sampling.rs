//! Seeded top-k/nucleus sampling and causal sample-and-match verification.
use anyhow::{Result, anyhow, ensure};

mod random;
mod speculative;

pub struct Distribution {
    entries: Vec<(u32, f64)>,
}

pub struct Sampling {
    temperature: f64,
    top_k: usize,
    top_p: f64,
    random: random::Random,
    verification_random: random::Random,
}

impl Sampling {
    pub fn settings(&self) -> (f64, usize, f64) {
        (self.temperature, self.top_k, self.top_p)
    }
    pub fn new(temperature: f64, top_k: usize, top_p: f64, seed: u64) -> Result<Self> {
        ensure!(
            temperature.is_finite() && temperature > 0.0,
            "temperature must be finite and positive"
        );
        ensure!((1..=20).contains(&top_k), "sampling top_k must be 1..=20");
        ensure!(
            top_p.is_finite() && top_p > 0.0 && top_p <= 1.0,
            "top_p must be in (0,1]"
        );
        Ok(Self {
            temperature,
            top_k,
            top_p,
            random: random::Random::new(seed as u32),
            // Keep acceptance/residual draws separate from target draws:
            // identical seeds would reuse words already revealed by bonuses.
            verification_random: random::Random::new(seed as u32 ^ 0x9e3779b9),
        })
    }

    pub fn distribution(
        &self,
        candidates: impl IntoIterator<Item = (u32, f32)>,
    ) -> Result<Distribution> {
        let mut candidates: Vec<_> = candidates.into_iter().collect();
        ensure!(
            candidates
                .iter()
                .all(|(_, x)| !x.is_nan() && *x != f32::INFINITY),
            "invalid sampling logits"
        );
        candidates.retain(|(_, x)| *x != f32::NEG_INFINITY);
        let order = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then(b.0.cmp(&a.0));
        if candidates.len() > self.top_k {
            candidates.select_nth_unstable_by(self.top_k, order);
        }
        candidates.truncate(self.top_k);
        candidates.sort_unstable_by(order);
        let max = candidates
            .first()
            .ok_or_else(|| anyhow!("empty sampling distribution"))?
            .1 as f64;
        for (i, (id, _)) in candidates.iter().enumerate() {
            ensure!(
                !candidates[..i].iter().any(|(other, _)| other == id),
                "duplicate sampling candidate"
            );
        }
        let mut entries: Vec<_> = candidates
            .into_iter()
            .map(|(id, logit)| (id, ((logit as f64 - max) / self.temperature).exp()))
            .collect();
        let total: f64 = entries.iter().map(|(_, p)| p).sum();
        let mut mass = 0.0;
        let mut keep = entries.len();
        for (i, (_, weight)) in entries.iter().enumerate() {
            mass += weight;
            if mass >= self.top_p * total {
                keep = i + 1;
                break;
            }
        }
        entries.truncate(keep);
        let total: f64 = entries.iter().map(|(_, p)| p).sum();
        for (_, p) in &mut entries {
            *p /= total;
        }
        Ok(Distribution { entries })
    }

    fn uniform(&mut self) -> f64 {
        self.random.uniform()
    }

    pub fn draw(&mut self, distribution: &Distribution) -> u32 {
        if distribution.entries.len() == 1 {
            return distribution.entries[0].0;
        }
        let sample = self.uniform();
        distribution.at(sample)
    }

    /// Greedy MTP proposals define a point-mass draft distribution. Drawing
    /// from each target row and accepting only a matching proposal preserves
    /// target sampling exactly; never draw from a row after the first mismatch.
    pub fn select_prefix(
        &mut self,
        distributions: &[Distribution],
        proposals: &[u32],
        eos: u32,
    ) -> Result<(usize, u32)> {
        ensure!(
            distributions.len() == proposals.len() + 1,
            "invalid verification distributions"
        );
        for (i, proposal) in proposals.iter().enumerate() {
            let token = self.draw(&distributions[i]);
            if token != *proposal {
                return Ok((i, token));
            }
            if token == eos {
                return Ok((i + 1, eos));
            }
        }
        Ok((proposals.len(), self.draw(&distributions[proposals.len()])))
    }
}

impl Distribution {
    fn at(&self, sample: f64) -> u32 {
        let mut cumulative = 0.0;
        for &(id, p) in &self.entries {
            cumulative += p;
            if sample < cumulative {
                return id;
            }
        }
        self.entries
            .last()
            .expect("nonempty normalized distribution")
            .0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_top_k_and_nucleus() -> Result<()> {
        let s = Sampling::new(2.0, 3, 1.0, 42)?;
        let d = s.distribution([(0, 0.0), (1, 2.0), (2, 4.0), (3, -100.0)])?;
        let z = 1.0 + (-1.0f64).exp() + (-2.0f64).exp();
        for ((id, p), (eid, ep)) in d.entries.iter().zip([
            (2, 1.0 / z),
            (1, (-1.0f64).exp() / z),
            (0, (-2.0f64).exp() / z),
        ]) {
            assert_eq!(*id, eid);
            assert!((*p - ep).abs() < 1e-14);
        }
        let s = Sampling::new(1.0, 20, 0.7, 0)?;
        let d = s.distribution([(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)])?;
        assert_eq!(d.entries.len(), 3);
        assert_eq!(
            d.entries.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            [3, 2, 1]
        );
        assert_eq!(d.at(0.0), 3);
        assert_eq!(d.at(1.0 / 3.0), 2);
        assert_eq!(d.at(0.99999), 1);
        Ok(())
    }

    #[test]
    fn invalid_and_extreme_distributions() -> Result<()> {
        assert!(Sampling::new(0.0, 20, 0.95, 0).is_err());
        assert!(Sampling::new(1.0, 21, 0.95, 0).is_err());
        assert!(Sampling::new(1.0, 20, 0.0, 0).is_err());
        let s = Sampling::new(f64::MIN_POSITIVE, 20, 1.0, 0)?;
        assert!(s.distribution([]).is_err());
        assert!(s.distribution([(0, f32::NAN)]).is_err());
        assert!(s.distribution([(0, f32::INFINITY)]).is_err());
        assert!(s.distribution([(0, f32::NEG_INFINITY)]).is_err());
        assert!(s.distribution([(1, 0.0), (1, 1.0)]).is_err());
        let d = s.distribution([(0, -f32::MAX), (1, f32::MAX)])?;
        assert_eq!(d.at(0.999999), 1);
        Ok(())
    }

    #[test]
    fn partial_selection_matches_full_sort() -> Result<()> {
        let logits: Vec<_> = (0..4860u32)
            .map(|id| {
                (
                    id,
                    match id % 41 {
                        0 => f32::NEG_INFINITY,
                        1 => -0.0,
                        2 => 0.0,
                        _ => ((id * 73 % 997) as f32 - 498.0) * 0.03,
                    },
                )
            })
            .collect();
        let mut sorted = logits.clone();
        sorted.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(b.0.cmp(&a.0)));
        for top_k in 1..=20 {
            for top_p in [0.01, 0.7, 0.95, 1.0] {
                let s = Sampling::new(1.0, top_k, top_p, 42)?;
                let actual = s.distribution(logits.iter().copied())?;
                let expected = s.distribution(sorted[..top_k].iter().copied())?;
                assert_eq!(actual.entries, expected.entries);
            }
        }
        Ok(())
    }

    #[test]
    fn prefix_matching_preserves_the_target_random_stream() -> Result<()> {
        let settings = [(0, [1, 2, 3]), (42, [2, 1, 0]), (2026, [3, 3, 3])];
        for (seed, proposals) in settings {
            let mut speculative = Sampling::new(1.0, 20, 1.0, seed)?;
            let mut reference = Sampling::new(1.0, 20, 1.0, seed)?;
            let distributions: Vec<_> = (0..4)
                .map(|i| speculative.distribution((0..4).map(|id| (id, (id + i) as f32 % 3.0))))
                .collect::<Result<_>>()?;
            let (accepted, next) = speculative.select_prefix(&distributions, &proposals, 100)?;
            for i in 0..accepted {
                assert_eq!(reference.draw(&distributions[i]), proposals[i]);
            }
            assert_eq!(reference.draw(&distributions[accepted]), next);
            assert_eq!(speculative.uniform(), reference.uniform());
        }
        let mut speculative = Sampling::new(1.0, 20, 1.0, 42)?;
        let mut reference = Sampling::new(1.0, 20, 1.0, 42)?;
        let distributions = [
            speculative.distribution([(9, 0.0)])?,
            speculative.distribution([(13, 0.0)])?,
        ];
        assert_eq!(speculative.select_prefix(&distributions, &[9], 9)?, (1, 9));
        assert_eq!(reference.draw(&distributions[0]), 9);
        assert_eq!(
            speculative.uniform(),
            reference.uniform(),
            "EOS consumed an unused suffix sample"
        );
        Ok(())
    }
}
