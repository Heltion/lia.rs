//! Rejection sampling for stochastic draft proposals, preserving the target law.
use super::*;

impl Sampling {
    pub fn select_speculative(
        &mut self,
        targets: &[Distribution],
        proposals: &[u32],
        drafts: &[Distribution],
        eos: u32,
    ) -> Result<(usize, u32)> {
        ensure!(
            !self.position_keyed(),
            "position-keyed Gumbel does not support stochastic draft rejection"
        );
        ensure!(
            targets.len() == proposals.len() + 1 && drafts.len() == proposals.len(),
            "invalid speculative distributions"
        );
        for (i, (&proposal, draft)) in proposals.iter().zip(drafts).enumerate() {
            let target = &targets[i];
            // The reference sampler advances its target draw stream for each
            // candidate independently of the acceptance/residual random stream.
            let fallback = self.draw(target);
            let p = target.probability(proposal);
            let q = draft.probability(proposal);
            ensure!(q > 0.0, "draft proposal has zero probability");
            if p >= q || (self.verification_random.uniform_f32() as f64) < p / q {
                if proposal == eos {
                    return Ok((i + 1, eos));
                }
                continue;
            }
            let residual: Vec<_> = target
                .entries
                .iter()
                .map(|&(id, p)| (id, (p - draft.probability(id)).max(0.0)))
                .filter(|&(_, p)| p > 0.0)
                .collect();
            let mass: f64 = residual.iter().map(|&(_, p)| p).sum();
            if mass == 0.0 {
                return Ok((i, fallback));
            }
            let mut u = self.verification_random.uniform_f32() as f64 * mass;
            for &(id, probability) in &residual {
                u -= probability;
                if u <= 0.0 {
                    return Ok((i, id));
                }
            }
            return Ok((i, residual.last().unwrap().0));
        }
        Ok((proposals.len(), self.draw(targets.last().unwrap())))
    }
}

impl Distribution {
    fn probability(&self, token: u32) -> f64 {
        self.entries
            .iter()
            .find(|&&(id, _)| id == token)
            .map_or(0.0, |&(_, p)| p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acceptance_is_independent_of_earlier_bonus_tokens() -> Result<()> {
        let mut counts = [[0usize; 2]; 2];
        for seed in 0..50_000 {
            let mut sampler = Sampling::new(1.0, 2, 1.0, seed)?;
            let point = || Distribution {
                entries: vec![(0, 1.0)],
            };
            let balanced = || Distribution {
                entries: vec![(0, 0.5), (1, 0.5)],
            };
            // First emit a fair bonus token without consuming acceptance RNG.
            let (_, previous) =
                sampler.select_speculative(&[point(), balanced()], &[0], &[point()], 99)?;
            // Condition on acceptance of an intervening proposal. The next
            // target must remain fair for either earlier bonus token.
            let (accepted, _) =
                sampler.select_speculative(&[balanced(), point()], &[0], &[point()], 99)?;
            if accepted == 0 {
                continue;
            }
            let (accepted, corrected) =
                sampler.select_speculative(&[balanced(), point()], &[0], &[point()], 99)?;
            let token = if accepted == 0 { corrected } else { 0 };
            counts[previous as usize][token as usize] += 1;
        }
        for row in counts {
            let total = row.iter().sum::<usize>();
            assert!(total > 10_000);
            assert!((row[0] as f64 / total as f64 - 0.5).abs() < 0.02);
        }
        Ok(())
    }

    #[test]
    fn acceptance_and_residual_reproduce_target_mass() -> Result<()> {
        let p = Distribution {
            entries: vec![(0, 0.7), (1, 0.2), (2, 0.1)],
        };
        let q = Distribution {
            entries: vec![(0, 0.1), (1, 0.7), (2, 0.2)],
        };
        let accepted: f64 = p
            .entries
            .iter()
            .map(|&(id, v)| v.min(q.probability(id)))
            .sum();
        let residual: f64 = p
            .entries
            .iter()
            .map(|&(id, v)| (v - q.probability(id)).max(0.0))
            .sum();
        assert!((accepted + residual - 1.0).abs() < 1e-14);
        for &(id, expected) in &p.entries {
            let actual = expected.min(q.probability(id))
                + (1.0 - accepted) * (expected - q.probability(id)).max(0.0) / residual;
            assert!((actual - expected).abs() < 1e-14);
        }
        let mut target = Sampling::new(1.0, 20, 1.0, 42)?;
        let mut draft = Sampling::new(1.0, 20, 1.0, 42 ^ 0x85ebca6b)?;
        let targets = [
            p,
            Distribution {
                entries: vec![(0, 1.0)],
            },
        ];
        let drafts = [q];
        let mut counts = [0usize; 3];
        for _ in 0..100_000 {
            let proposal = draft.draw(&drafts[0]);
            let (accepted, corrected) =
                target.select_speculative(&targets, &[proposal], &drafts, 100)?;
            counts[if accepted == 0 { corrected } else { proposal } as usize] += 1;
        }
        for (count, expected) in counts.into_iter().zip([0.7, 0.2, 0.1]) {
            assert!((count as f64 / 100_000.0 - expected).abs() < 0.005);
        }
        Ok(())
    }

    #[test]
    fn zero_support_rejection_and_eos_stop_causally() -> Result<()> {
        let mut target = Sampling::new(1.0, 20, 1.0, 42)?;
        let p = [
            Distribution {
                entries: vec![(2, 1.0)],
            },
            Distribution {
                entries: vec![(3, 1.0)],
            },
        ];
        let q = [Distribution {
            entries: vec![(1, 1.0)],
        }];
        assert_eq!(target.select_speculative(&p, &[1], &q, 99)?, (0, 2));
        let eos = [
            Distribution {
                entries: vec![(9, 1.0)],
            },
            Distribution {
                entries: vec![(13, 1.0)],
            },
        ];
        let drafts = [Distribution {
            entries: vec![(9, 1.0)],
        }];
        let mut reference = Sampling::new(1.0, 20, 1.0, 42)?;
        assert_eq!(reference.draw(&eos[0]), 9);
        let mut sampler = Sampling::new(1.0, 20, 1.0, 42)?;
        assert_eq!(sampler.select_speculative(&eos, &[9], &drafts, 9)?, (1, 9));
        assert_eq!(sampler.uniform(), reference.uniform());
        Ok(())
    }
}
