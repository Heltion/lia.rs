//! Full-vocabulary mass controls nucleus selection, even when candidates are sparse.
use super::*;

fn order(a: &(u32, f32), b: &(u32, f32)) -> std::cmp::Ordering {
    b.1.total_cmp(&a.1).then(b.0.cmp(&a.0))
}

impl Sampling {
    /// Dense IDs are unique by construction; avoid hashing and sorting the whole vocabulary.
    pub fn distribution_logits(&self, logits: &[f32]) -> Result<Distribution> {
        if self.top_k != 0 || self.top_p == 1.0 {
            return self.distribution(
                logits
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(i, v)| (i as u32, v)),
            );
        }
        ensure!(
            logits.len() <= u32::MAX as usize,
            "sampling vocabulary exceeds token ID range"
        );
        ensure!(
            logits.iter().all(|v| !v.is_nan() && *v != f32::INFINITY),
            "invalid sampling logits"
        );
        let mut candidates: Vec<_> = logits
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, v)| *v != f32::NEG_INFINITY)
            .map(|(i, v)| (i as u32, v))
            .collect();
        let maximum = candidates
            .iter()
            .map(|(_, v)| *v)
            .max_by(f32::total_cmp)
            .context("empty sampling distribution")? as f64;
        let total: f64 = candidates
            .iter()
            .map(|(_, v)| ((f64::from(*v) - maximum) / self.temperature).exp())
            .sum();
        let entries = self.select_nucleus(&mut candidates, maximum, total)?;
        Ok(Distribution { entries })
    }

    /// Accept sparse GPU candidates only when both the cutoff and omitted tail are proven safe.
    pub(crate) fn checked_nucleus(
        &self,
        mut candidates: Vec<(u32, f32)>,
        maximum: f64,
        total: f64,
        omitted: f32,
    ) -> Result<Option<Distribution>> {
        ensure!(
            total.is_finite() && total > 0.0 && maximum.is_finite(),
            "invalid full-vocabulary normalizer"
        );
        if self.top_p == 1.0 {
            return Ok(None);
        }
        let entries = self.select_nucleus(&mut candidates, maximum, total)?;
        let Some(&(last, _)) = entries.last() else {
            return Ok(None);
        };
        let keep = entries.len();
        // select_nucleus leaves its retained prefix sorted by the complete token key.
        ensure!(
            candidates[keep - 1].0 == last,
            "inconsistent nucleus candidate order"
        );
        if candidates[keep - 1].1 <= omitted {
            return Ok(None);
        }
        let target = self.top_p * total;
        let through: f64 = candidates[..keep]
            .iter()
            .map(|(_, v)| ((f64::from(*v) - maximum) / self.temperature).exp())
            .sum();
        let before =
            through - ((f64::from(candidates[keep - 1].1) - maximum) / self.temperature).exp();
        // GPU FP64 reductions and CPU sorted reductions differ in their addition order.
        // An ambiguous boundary falls back to the complete CPU distribution.
        let uncertainty = total.abs() * 1e-10;
        if before >= target - uncertainty || through <= target + uncertainty {
            return Ok(None);
        }
        Ok(Some(Distribution { entries }))
    }

    fn select_nucleus(
        &self,
        candidates: &mut [(u32, f32)],
        maximum: f64,
        total: f64,
    ) -> Result<Vec<(u32, f64)>> {
        ensure!(!candidates.is_empty(), "empty nucleus candidate list");
        let weight = |v: f32| ((f64::from(v) - maximum) / self.temperature).exp();
        let target = self.top_p * total;
        let mut count = candidates.len().min(128);
        loop {
            if count < candidates.len() {
                candidates.select_nth_unstable_by(count, order);
            }
            let mass: f64 = candidates[..count].iter().map(|(_, v)| weight(*v)).sum();
            if mass >= target || count == candidates.len() {
                break;
            }
            count = (count * 2).min(candidates.len());
        }
        candidates[..count].sort_unstable_by(order);
        let mut mass = 0.0;
        let mut entries = Vec::new();
        for &(id, logit) in &candidates[..count] {
            let value = weight(logit);
            mass += value;
            entries.push((id, value));
            if mass >= target {
                break;
            }
        }
        ensure!(mass.is_finite() && mass > 0.0, "invalid nucleus mass");
        for (_, p) in &mut entries {
            *p /= mass;
        }
        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partial_selection_matches_independent_full_sort() -> Result<()> {
        for temperature in [0.2, 1.0, 2.0] {
            for top_p in [0.01, 0.5, 0.95, 1.0] {
                let sampling = Sampling::new(temperature, 0, top_p, 42)?;
                let logits: Vec<_> = (0..4096)
                    .map(|i| ((i * 173 % 4096) as f32 - 2048.0) / 64.0)
                    .collect();
                let expected = sampling.distribution(
                    logits
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(i, v)| (i as u32, v)),
                )?;
                let actual = sampling.distribution_logits(&logits)?;
                assert_eq!(actual.entries.len(), expected.entries.len());
                for (a, b) in actual.entries.iter().zip(&expected.entries) {
                    assert_eq!(a.0, b.0);
                    assert!((a.1 - b.1).abs() < 1e-12);
                }
            }
        }
        Ok(())
    }
    #[test]
    fn omitted_high_scores_and_ambiguous_boundaries_force_fallback() -> Result<()> {
        let sampling = Sampling::new(1.0, 0, 0.5, 42)?;
        assert!(
            sampling
                .checked_nucleus(vec![(0, 0.0), (1, -1.0)], 0.0, 2.0, 0.1)?
                .is_none()
        );
        assert!(
            sampling
                .checked_nucleus(vec![(0, 0.0), (1, -1.0)], 0.0, 2.0, -10.0)?
                .is_none()
        );
        assert!(
            sampling
                .checked_nucleus(vec![(0, 0.0), (1, -1.0)], 0.0, 1.5, -10.0)?
                .is_some()
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA device"]
    fn gpu_candidates_keep_full_nucleus_mass_and_safe_fallbacks() -> Result<()> {
        use cuda_core::{CudaContext, DeviceBuffer};
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = crate::gpu::kernels::load(&context)?;
        let vocabulary = 3073;
        let mut logits = Vec::new();
        logits.extend(vec![0_f32; vocabulary]);
        logits.extend((0..vocabulary).map(|id| if id == 2048 { 20.0 } else { -20.0 }));
        logits.extend((0..vocabulary).map(|id| if id < 300 { 1.0 } else { -20.0 }));
        logits.extend((0..vocabulary).map(|id| {
            if id % 1024 < 4 {
                1.0 - (id % 1024) as f32 * 0.2 + (id / 1024) as f32 * 0.03
            } else {
                -50.0
            }
        }));
        let device = DeviceBuffer::from_host(&stream, &logits)?;
        let mut workspace = crate::gpu::nucleus::Workspace::new(&stream, vocabulary, 4)?;
        let temperatures = [1.0, 1.0, 1.0, 0.7];
        workspace.prepare(&stream, &temperatures)?;
        let rows = workspace.read(&module, &stream, &device)?;
        for (index, row) in rows.into_iter().enumerate() {
            let sampling = Sampling::new(temperatures[index], 0, 0.95, 42)?;
            let expected = sampling.distribution(
                logits[index * vocabulary..(index + 1) * vocabulary]
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(i, v)| (i as u32, v)),
            )?;
            let sparse =
                sampling.checked_nucleus(row.candidates, row.maximum, row.total, row.omitted)?;
            assert_eq!(sparse.is_some(), [false, true, false, true][index]);
            let actual = sparse.unwrap_or(
                sampling
                    .distribution_logits(&logits[index * vocabulary..(index + 1) * vocabulary])?,
            );
            assert_eq!(actual.entries.len(), expected.entries.len());
            for (a, b) in actual.entries.iter().zip(&expected.entries) {
                assert_eq!(a.0, b.0);
                assert!((a.1 - b.1).abs() < 1e-12);
            }
        }
        logits[37] = f32::NAN;
        let invalid = DeviceBuffer::from_host(&stream, &logits)?;
        assert!(workspace.read(&module, &stream, &invalid).is_err());
        Ok(())
    }
}
