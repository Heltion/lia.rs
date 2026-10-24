#![cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};
use lia::models::minicpm_o45::audio::{Features, stream::Streaming};
use serde_json::Value;
use std::path::Path;

fn read(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(bytes.len().is_multiple_of(4), "truncated Float32 tensor");
    Ok(bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
#[test]
#[ignore = "requires official streaming Whisper fixtures, checkpoint and CUDA"]
fn cached_whisper_matches_official_across_normalization_and_position_reset() -> Result<()> {
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_STREAMING_WHISPER_ORACLE")?);
    let info: Value = serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let mut engine = lia::inference::Engine::load_huggingface(&model, 64, false)?;
    engine.load_audio(&model)?;
    let mut failures = 0;
    for group in info["groups"]
        .as_array()
        .context("missing streaming Whisper groups")?
    {
        let mut direct = engine.new_audio_cache()?;
        let mut native = engine.new_audio_cache()?;
        let mut processor = Streaming::new(1000, 1035, 20, group["sliding"] == true)?;
        ensure!(
            engine
                .audio_stream_embeddings(
                    &mut direct,
                    &Features {
                        frames: 0,
                        values: Vec::new()
                    },
                    0,
                    0
                )
                .is_err()
                && direct.position() == 0,
            "invalid input advanced audio cache"
        );
        for (index, chunk) in group["chunks"]
            .as_array()
            .context("missing streaming Whisper chunks")?
            .iter()
            .enumerate()
        {
            let frames = chunk["frames"].as_u64().context("missing mel frames")? as usize;
            let features = Features {
                frames,
                values: read(Path::new(
                    chunk["mel"].as_str().context("missing mel path")?,
                ))?,
            };
            let samples = read(Path::new(
                chunk["input"].as_str().context("missing PCM path")?,
            ))?;
            let native_features = processor
                .push(&samples, group["name"] == "short-final" && index == 1)?
                .features;
            let expected =
                read(&root.join(chunk["features"].as_str().context("missing embeddings")?))?;
            let prefix = chunk["prefix"].as_u64().context("missing CNN prefix")? as usize;
            let suffix = chunk["suffix"].as_u64().context("missing CNN suffix")? as usize;
            for (name, cache, features) in [
                ("official-mel", &mut direct, &features),
                ("native-pcm", &mut native, &native_features),
            ] {
                let actual = engine.audio_stream_embeddings(cache, features, prefix, suffix)?;
                ensure!(
                    actual.len() == expected.len()
                        && actual.iter().chain(&expected).all(|v| v.is_finite()),
                    "invalid streaming embedding extent/value"
                );
                let maximum = actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0., f32::max);
                let rms = (actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| f64::from(a - b).powi(2))
                    .sum::<f64>()
                    / actual.len() as f64)
                    .sqrt();
                println!(
                    "{} chunk {index} {name}: max_abs={maximum}, rms={rms}, cache_frames={}",
                    group["name"],
                    cache.position()
                );
                failures += usize::from(maximum >= 0.03 || rms >= 0.005);
                ensure!(
                    cache.position()
                        == chunk["cache_frames"]
                            .as_u64()
                            .context("missing cache length")? as usize,
                    "audio cache reset/position mismatch"
                );
            }
        }
    }
    ensure!(
        failures == 0,
        "{failures} streaming Whisper comparisons exceed existing full-encoder gates"
    );
    Ok(())
}
