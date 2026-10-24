#![cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::path::Path;
fn read(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(
        bytes.len().is_multiple_of(4),
        "truncated Float32 embedding fixture"
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
#[test]
#[ignore = "requires checkpoint, official mixed-unit Qwen3 oracle and CUDA"]
fn mixed_audio_embeddings_and_tokens_share_causal_language_history() -> Result<()> {
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_DUPLEX_FEED_ORACLE")?);
    let info: Value = serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let mut engine = lia::inference::Engine::load_huggingface(&model, 128, false)?;
    engine.load_audio(&model)?;
    ensure!(
        engine.forward_embeddings(&[f32::NAN; 4096], false).is_err()
            && engine.forward_embeddings(&[0.; 4095], false).is_err()
            && engine.position() == 0,
        "invalid direct input advanced language cache"
    );
    let segments = info["segments"]
        .as_array()
        .context("missing feed segments")?;
    for native_pcm in [false, true] {
        engine.reset()?;
        let mut audio = engine.new_audio_cache()?;
        let mut processor =
            lia::models::minicpm_o45::audio::stream::Streaming::new(1000, 1035, 20, false)?;
        let mut position = 0;
        let mut logits = Vec::new();
        for (index, segment) in segments.iter().enumerate() {
            let want = index + 1 == segments.len();
            if let Some(path) = segment["embeddings"].as_str() {
                let values = if native_pcm {
                    let samples = read(Path::new(
                        segment["pcm"].as_str().context("missing PCM fixture")?,
                    ))?;
                    let features = processor.push(&samples, false)?.features;
                    engine.audio_stream_embeddings(
                        &mut audio,
                        &features,
                        segment["prefix_frames"]
                            .as_u64()
                            .context("missing CNN prefix")? as usize,
                        segment["suffix_frames"]
                            .as_u64()
                            .context("missing CNN suffix")? as usize,
                    )?
                } else {
                    read(Path::new(path))?
                };
                position += values.len() / 4096;
                logits = engine.forward_embeddings(&values, true)?;
                ensure!(
                    logits.len() == values.len() / 4096 * engine.vocabulary_size()
                        && logits.iter().all(|v| v.is_finite()),
                    "invalid direct embedding logits"
                );
            } else {
                let tokens: Vec<u32> = serde_json::from_value(segment["tokens"].clone())?;
                position += tokens.len();
                logits = engine.forward_many(&tokens, want)?;
            }
            ensure!(
                engine.position() == position,
                "mixed feed lost language positions"
            );
        }
        let last = &logits[logits.len() - engine.vocabulary_size()..];
        let expected = read(&root.join("last-logits.f32"))?;
        ensure!(
            last.len() == expected.len() && last.iter().chain(&expected).all(|v| v.is_finite()),
            "invalid mixed-feed logits"
        );
        let maximum = last
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f32::max);
        let best = |values: &[f32]| {
            values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        println!(
            "mixed token/audio units native_pcm={native_pcm}: position={position}, max_abs={maximum}, native_top={}, reference_top={}",
            best(last),
            best(&expected)
        );
        ensure!(
            maximum < 0.025 && best(last) == best(&expected),
            "mixed feed reference mismatch"
        );
        let saved = engine.snapshot()?;
        engine.forward(151683)?;
        engine.restore(&saved)?;
        ensure!(
            engine.position() == position,
            "mixed feed snapshot failed to restore"
        );
    }
    Ok(())
}
