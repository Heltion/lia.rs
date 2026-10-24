//! Stable categorical draws and checkpoint top-k/nucleus filtering.
use anyhow::{Result, ensure};
pub(super) fn argmax(values: &[f32]) -> Result<u32> {
    let mut best = None;
    for (i, &value) in values.iter().enumerate() {
        if value.is_finite() && best.is_none_or(|(_, v)| value > v) {
            best = Some((i, value));
        }
    }
    Ok(best
        .ok_or_else(|| anyhow::anyhow!("empty duplex distribution"))?
        .0 as u32)
}
pub(super) fn probabilities(scores: &[f32]) -> Result<Vec<f64>> {
    let maximum = scores[argmax(scores)? as usize] as f64;
    let mut values: Vec<f64> = scores
        .iter()
        .map(|v| (f64::from(*v) - maximum).exp())
        .collect();
    let sum: f64 = values.iter().sum();
    ensure!(sum.is_finite() && sum > 0., "invalid duplex normalizer");
    for v in &mut values {
        *v /= sum;
    }
    Ok(values)
}
pub(super) fn categorical(
    probabilities: &[f64],
    draw: &mut impl FnMut(&[f64]) -> Result<f64>,
) -> Result<u32> {
    let value = draw(probabilities)?;
    ensure!(
        value.is_finite() && (0.0..1.0).contains(&value),
        "invalid duplex uniform draw"
    );
    let mut sum = 0.;
    let mut last = 0;
    for (id, &p) in probabilities.iter().enumerate() {
        if p > 0. {
            last = id;
            sum += p;
            if value < sum {
                return Ok(id as u32);
            }
        }
    }
    Ok(last as u32)
}
pub(super) fn filter(scores: &mut [f32], k: usize, p: f64) -> Result<()> {
    ensure!(
        scores.iter().all(|v| !v.is_nan() && *v != f32::INFINITY),
        "duplex temperature overflow"
    );
    let mut candidates: Vec<(usize, f32)> = scores
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .collect();
    ensure!(!candidates.is_empty(), "empty duplex text distribution");
    if k > 0 && candidates.len() > k {
        candidates.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
        let cutoff = candidates[k - 1].1;
        candidates.retain(|(_, v)| *v >= cutoff);
    }
    candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    if p < 1. {
        let maximum = f64::from(candidates[0].1);
        let sum: f64 = candidates
            .iter()
            .map(|(_, v)| (f64::from(*v) - maximum).exp())
            .sum();
        let mut mass = 0.;
        let mut keep = candidates.len();
        for (i, (_, value)) in candidates.iter().enumerate() {
            mass += (f64::from(*value) - maximum).exp();
            if mass > p * sum {
                keep = i + 1;
                break;
            }
        }
        candidates.truncate(keep);
    }
    scores.fill(f32::NEG_INFINITY);
    for (id, value) in candidates {
        scores[id] = value;
    }
    Ok(())
}
