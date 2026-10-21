//! Position-indexed Gumbel-max noise using the public-domain MurmurHash3 mixer.
//! MurmurHash3: Austin Appleby, https://github.com/aappleby/smhasher.

fn hash(seed: u64, position: u64, token: u32) -> u32 {
    let mut h = 0u32;
    for word in [seed as u32, (seed >> 32) as u32, position as u32, token] {
        let k = word
            .wrapping_mul(0xcc9e2d51)
            .rotate_left(15)
            .wrapping_mul(0x1b873593);
        h = (h ^ k)
            .rotate_left(13)
            .wrapping_mul(5)
            .wrapping_add(0xe6546b64);
    }
    h ^= 16;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2ae35);
    h ^ (h >> 16)
}

fn uniform(word: u32) -> f32 {
    (word >> 16) as f32 * (1.0 / 65536.0) + ((word & 65535) as f32 + 0.5) * (1.0 / 4294967296.0)
}

fn noise(uniform: f32) -> f32 {
    // Evaluate -log(1-u) without cancellation in the winning small-u tail.
    let negative_log = if uniform < 0.25 {
        let mut polynomial = 1.0f32 / 8.0;
        for denominator in (1..8).rev() {
            polynomial = uniform.mul_add(polynomial, 1.0 / denominator as f32);
        }
        uniform * polynomial
    } else {
        -(1.0 - uniform).max(1.0 / 16777216.0).ln()
    };
    -negative_log.ln()
}

pub(super) fn draw(entries: &[(u32, f64)], seed: u64, position: u64) -> u32 {
    entries
        .iter()
        .map(|&(token, probability)| {
            let score = probability.ln() as f32 + noise(uniform(hash(seed, position, token)));
            (token, score)
        })
        .max_by(|a, b| a.1.total_cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
        .expect("nonempty normalized distribution")
        .0
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context, Result, ensure};

    #[test]
    #[ignore = "requires LIA_TEST_NOISE_REFERENCE with external GPU reference vectors"]
    fn mapping_matches_gpu_reference() -> Result<()> {
        let path =
            std::env::var("LIA_TEST_NOISE_REFERENCE").context("set LIA_TEST_NOISE_REFERENCE")?;
        let data: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&path).with_context(|| format!("reading {path}"))?,
        )
        .context("parsing GPU random reference")?;
        let rows = data["records"]
            .as_array()
            .context("missing reference records")?;
        ensure!(!rows.is_empty(), "empty GPU reference");
        let mut maximum = 0.0f32;
        for row in rows {
            let seed = row["seed"].as_u64().context("invalid reference seed")?;
            let position = row["position"]
                .as_u64()
                .context("invalid reference position")?;
            let token = u32::try_from(row["token"].as_u64().context("invalid reference token")?)?;
            let word = hash(seed, position, token);
            ensure!(
                Some(u64::from(word)) == row["hash"].as_u64(),
                "hash differs"
            );
            let actual = uniform(word);
            let expected = row["uniform"]
                .as_f64()
                .context("invalid uniform reference")? as f32;
            ensure!(actual.to_bits() == expected.to_bits(), "uniform differs");
            let expected_noise = row["noise"].as_f64().context("invalid noise reference")? as f32;
            maximum = maximum.max((noise(actual) - expected_noise).abs());
        }
        eprintln!(
            "GPU random mapping rows={} maximum_noise_error={maximum}",
            rows.len()
        );
        ensure!(maximum < 1e-5, "noise exceeds floating point tolerance");
        Ok(())
    }

    #[test]
    fn gumbel_max_reproduces_categorical_mass() {
        let entries = [(0, 0.2), (1, 0.3), (2, 0.5)];
        let mut counts = [0u32; 3];
        for seed in 0..30_000 {
            counts[draw(&entries, seed, 2050) as usize] += 1;
        }
        for (count, expected) in counts.into_iter().zip([0.2, 0.3, 0.5]) {
            assert!((f64::from(count) / 30_000.0 - expected).abs() < 0.015);
        }
    }
}
