use anyhow::{Context, Result, ensure};
use lia::sampling::speech::{Parameters, Policy, Sampler};
use serde_json::Value;
use std::path::Path;
fn read(root: &Path, name: &Value) -> Result<Vec<f32>> {
    let bytes = std::fs::read(root.join(name.as_str().context("missing chunk sampling fixture")?))?;
    ensure!(
        bytes.len().is_multiple_of(4),
        "truncated Float32 probabilities"
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
#[test]
#[ignore = "requires actual official generate_chunk probability records"]
fn chunk_policy_matches_official_probabilities_without_whole_utterance_filters() -> Result<()> {
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_SPEECH_CHUNK_ORACLE")?);
    let info: Value = serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    for group in info["groups"].as_array().context("missing chunk groups")? {
        for (index, chunk) in group["chunks"]
            .as_array()
            .context("missing chunks")?
            .iter()
            .enumerate()
        {
            let sampler = Sampler::new(
                Parameters {
                    policy: Policy::Chunk,
                    min_tokens: chunk["minimum"].as_u64().context("missing minimum")? as usize,
                    ..Parameters::default()
                },
                42,
            )?;
            let mut history = Vec::new();
            let mut maximum = 0.0f64;
            for draw in chunk["draws"]
                .as_array()
                .context("missing probability records")?
            {
                let actual = sampler.probabilities(&read(&root, &draw["logits"])?, &history)?;
                let expected = read(&root, &draw["probabilities"])?;
                ensure!(
                    actual.len() == expected.len(),
                    "chunk probability extent mismatch"
                );
                for (a, b) in actual.iter().zip(expected) {
                    maximum = maximum.max((a - f64::from(b)).abs());
                }
                let token = draw["token"].as_u64().context("missing sampled code")? as u32;
                if token != 6561 {
                    history.push(token);
                }
            }
            println!(
                "{} chunk {index}: probability_max_abs={maximum}",
                group["name"]
            );
            ensure!(maximum < 2e-6, "chunk policy probability mismatch");
        }
    }
    Ok(())
}
