#![cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};
#[test]
#[ignore = "requires official chunk encoder fixtures, checkpoint and CUDA"]
fn cached_encoder_chunks_match_official_reference() -> Result<()> {
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_CODEC_STREAM_ORACLE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let mut engine = lia::inference::Engine::load_huggingface(&model, 64, false)?;
    engine.load_codec_encoder(&model)?;
    for group in info["groups"].as_array().context("missing stream groups")? {
        let mut cache = engine.new_codec_encoder_cache()?;
        ensure!(
            cache.position() == 0 && !cache.finished(),
            "fresh encoder cache is not empty"
        );
        ensure!(
            engine
                .codec_encoder_chunk(&mut cache, &[1, 2, 3, 4], false)
                .is_err(),
            "nonfinal lookahead-only chunk accepted"
        );
        ensure!(
            engine
                .codec_encoder_chunk(&mut cache, &[6561], true)
                .is_err(),
            "invalid speech code accepted"
        );
        ensure!(
            cache.position() == 0 && !cache.finished(),
            "invalid input advanced encoder cache"
        );
        for (index, chunk) in group["chunks"]
            .as_array()
            .context("missing stream chunks")?
            .iter()
            .enumerate()
        {
            let codes: Vec<u32> = serde_json::from_value(chunk["codes"].clone())?;
            let last = chunk["last"].as_bool().context("missing final flag")?;
            let actual = engine.codec_encoder_chunk(&mut cache, &codes, last)?;
            let bytes = std::fs::read(
                root.join(
                    chunk["features"]
                        .as_str()
                        .context("missing stream features")?,
                ),
            )?;
            ensure!(
                actual.len() * 4 == bytes.len()
                    && actual.len()
                        == chunk["frames"].as_u64().context("missing frame count")? as usize * 80,
                "chunk output extent mismatch"
            );
            if let Some(suffix) = group["trim_suffix_frames"].as_u64() {
                engine.trim_codec_encoder_cache(
                    &mut cache,
                    group["prefix_frames"]
                        .as_u64()
                        .context("missing retained prefix")? as usize,
                    suffix as usize,
                )?;
                ensure!(
                    cache.history_frames()
                        == chunk["cached_frames"]
                            .as_u64()
                            .context("missing cached frame count")?
                            as usize,
                    "encoder retained history mismatch"
                );
                ensure!(
                    cache.history_codes() * 2 == cache.history_frames(),
                    "encoder rate histories diverged"
                );
            }
            let expected: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            ensure!(
                actual.iter().chain(&expected).all(|v| v.is_finite()),
                "nonfinite stream encoder output"
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
                "{} chunk {index}: max_abs={maximum}, rms={rms}, position={}",
                group["name"],
                cache.position()
            );
            ensure!(
                maximum < 0.01 && rms < 0.002,
                "cached chunk encoder mismatch"
            );
            ensure!(
                cache.position()
                    == chunk["position"].as_u64().context("missing position")? as usize
                    && cache.finished() == last,
                "stream encoder state mismatch"
            );
        }
        ensure!(
            engine.codec_encoder_chunk(&mut cache, &[1], true).is_err(),
            "finished encoder cache reused"
        );
    }
    Ok(())
}
