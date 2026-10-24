#![cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};
use lia::sampling::speech::{Parameters, Policy, Sampler};
use serde_json::Value;
use std::path::Path;
fn read(root: &Path, name: &Value) -> Result<Vec<f32>> {
    let bytes = std::fs::read(root.join(name.as_str().context("missing speech chunk tensor")?))?;
    ensure!(
        bytes.len().is_multiple_of(4),
        "truncated speech chunk tensor"
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
#[test]
#[ignore = "requires official retained-KV speech chunk oracle, checkpoint and CUDA"]
fn retained_speech_kv_and_unforwarded_lookahead_match_official_chunks() -> Result<()> {
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_SPEECH_CHUNK_ORACLE")?);
    let info: Value = serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let mut engine = lia::inference::Engine::load_huggingface(&model, 64, false)?;
    engine.load_speech(&model)?;
    for group in info["groups"].as_array().context("missing speech groups")? {
        let mut cache = engine.new_speech_cache(512)?;
        let seed = group["seed"].as_u64().context("missing speech seed")?;
        let mut sampler = Sampler::new(
            Parameters {
                policy: Policy::Chunk,
                min_tokens: 0,
                ..Parameters::default()
            },
            seed,
        )?;
        ensure!(
            engine
                .speech_generate_chunk(&mut cache, &[0.; 767], &mut sampler, 26)
                .is_err()
                && cache.position() == 0,
            "invalid chunk advanced speech cache"
        );
        for (index, chunk) in group["chunks"]
            .as_array()
            .context("missing speech chunks")?
            .iter()
            .enumerate()
        {
            let tokens: Vec<u32> = serde_json::from_value(chunk["tokens"].clone())?;
            let hidden = read(&root, &chunk["hidden"])?;
            let condition = engine.duplex_speech_conditioning(&tokens, &hidden)?;
            let expected = read(&root, &chunk["conditioning"])?;
            ensure!(
                condition.len() == (tokens.len() + 1) * 768 && condition.len() == expected.len(),
                "duplex conditioning terminator/extent mismatch"
            );
            let maximum = condition
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f32::max);
            ensure!(
                maximum < 0.002,
                "duplex semantic/text conditioning mismatch"
            );
            sampler
                .set_min_tokens(chunk["minimum"].as_u64().context("missing minimum")? as usize)?;
            let generated = engine.speech_generate_chunk(
                &mut cache,
                &condition,
                &mut sampler,
                chunk["maximum"].as_u64().context("missing maximum")? as usize,
            )?;
            println!(
                "{} chunk {index}: conditioning_max_abs={maximum}, codes={}, sampled={}, finished={}, cache_position={}",
                group["name"],
                generated.codes.len(),
                generated.sampled_tokens,
                generated.finished,
                cache.position()
            );
            ensure!(
                serde_json::to_value(&generated.codes)? == chunk["codes"]
                    && chunk["finished"] == generated.finished
                    && chunk["sampled_tokens"].as_u64() == Some(generated.sampled_tokens as u64)
                    && generated.sampled_tokens == generated.codes.len() + 1
                    && chunk["position"].as_u64() == Some(cache.position() as u64),
                "speech chunk codes/EOS/retained KV mismatch"
            );
        }
        cache.reset();
        ensure!(cache.position() == 0, "speech cache reset failed");
        let first = &group["chunks"][0];
        let tokens: Vec<u32> = serde_json::from_value(first["tokens"].clone())?;
        let condition =
            engine.duplex_speech_conditioning(&tokens, &read(&root, &first["hidden"])?)?;
        let mut replay_sampler = Sampler::new(
            Parameters {
                policy: Policy::Chunk,
                min_tokens: first["minimum"]
                    .as_u64()
                    .context("missing replay minimum")? as usize,
                ..Parameters::default()
            },
            seed,
        )?;
        let replay = engine.speech_generate_chunk(
            &mut cache,
            &condition,
            &mut replay_sampler,
            first["maximum"]
                .as_u64()
                .context("missing replay maximum")? as usize,
        )?;
        ensure!(
            serde_json::to_value(&replay.codes)? == first["codes"]
                && first["finished"] == replay.finished
                && first["position"].as_u64() == Some(cache.position() as u64),
            "speech cache reset retained stale history"
        );
    }
    Ok(())
}
