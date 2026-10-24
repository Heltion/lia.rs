//! Seeded Gaussian draws, isolated from token sampling's random stream.
use super::random::Random;
pub(crate) fn gaussian(seed: u64, count: usize) -> Vec<f32> {
    let mut random = Random::new((seed ^ (seed >> 32)) as u32);
    let mut values = Vec::with_capacity(count);
    while values.len() < count {
        let radius = (-2.0 * (1.0 - random.uniform()).ln()).sqrt();
        let angle = core::f64::consts::TAU * random.uniform();
        let (sine, cosine) = angle.sin_cos();
        values.push((radius * cosine) as f32);
        if values.len() < count {
            values.push((radius * sine) as f32);
        }
    }
    values
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn seeded_normals_have_finite_unit_variance_and_independent_seeds() {
        let x = gaussian(10427, 100_000);
        assert!(x.iter().all(|v| v.is_finite()));
        let mean = x.iter().map(|v| f64::from(*v)).sum::<f64>() / x.len() as f64;
        let variance = x
            .iter()
            .map(|v| (f64::from(*v) - mean).powi(2))
            .sum::<f64>()
            / x.len() as f64;
        assert!(
            mean.abs() < 0.01 && (variance - 1.0).abs() < 0.03,
            "mean={mean} variance={variance}"
        );
        assert_eq!(&x[..17], gaussian(10427, 17));
        assert_ne!(gaussian(10427, 17), gaussian(10428, 17));
        assert!(gaussian(0, 0).is_empty());
    }
}
