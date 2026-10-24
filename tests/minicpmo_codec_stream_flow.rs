#![cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};
use std::path::Path;
fn read(root: &Path, file: &serde_json::Value) -> Result<Vec<f32>> {
    let bytes = std::fs::read(root.join(file.as_str().context("missing flow fixture")?))?;
    ensure!(
        bytes.len().is_multiple_of(4),
        "invalid Float32 flow fixture"
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
#[test]
#[ignore = "requires official chunk CFM fixtures, checkpoint and CUDA"]
fn cached_flow_chunks_match_official_reference() -> Result<()> {
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_CODEC_FLOW_STREAM_ORACLE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let mut engine = lia::inference::Engine::load_huggingface(&model, 64, false)?;
    engine.load_codec_flow(&model)?;
    ensure!(
        engine.new_codec_flow_cache(0).is_err(),
        "zero-step flow cache accepted"
    );
    for group in info["groups"]
        .as_array()
        .context("missing flow stream groups")?
    {
        let steps = group["steps"].as_u64().context("missing step count")? as usize;
        let mut cache = engine.new_codec_flow_cache(steps)?;
        ensure!(
            cache.position() == 0 && cache.steps() == steps,
            "invalid initial flow cache"
        );
        ensure!(
            engine
                .codec_flow_chunk(&mut cache, &[], &[], &[0.; 80], &[])
                .is_err(),
            "empty flow chunk accepted"
        );
        ensure!(
            engine
                .codec_flow_chunk(&mut cache, &[f32::NAN; 80], &[0.; 80], &[0.; 80], &[0.; 80])
                .is_err(),
            "nonfinite flow chunk accepted"
        );
        ensure!(cache.position() == 0, "invalid input advanced flow cache");
        for (index, chunk) in group["chunks"]
            .as_array()
            .context("missing flow chunks")?
            .iter()
            .enumerate()
        {
            let noise = read(&root, &chunk["noise"])?;
            let mu = read(&root, &chunk["mu"])?;
            let speaker = read(&root, &chunk["speaker"])?;
            let condition = read(&root, &chunk["condition"])?;
            let actual = engine.codec_flow_chunk(&mut cache, &noise, &mu, &speaker, &condition)?;
            if let Some(suffix) = group["trim_suffix_frames"].as_u64() {
                engine.trim_codec_flow_cache(
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
                            .context("missing retained flow frames")?
                            as usize,
                    "flow retained history mismatch"
                );
            }
            let expected = read(&root, &chunk["mel"])?;
            ensure!(
                actual.len() == expected.len()
                    && actual.len()
                        == chunk["frames"].as_u64().context("missing frame count")? as usize * 80,
                "flow chunk extent mismatch"
            );
            ensure!(
                actual.iter().chain(&expected).all(|v| v.is_finite()),
                "nonfinite flow result"
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
            ensure!(maximum < 0.02 && rms < 0.004, "cached CFM chunk mismatch");
            ensure!(
                cache.position()
                    == chunk["position"]
                        .as_u64()
                        .context("missing flow position")? as usize,
                "flow cache position mismatch"
            );
        }
    }
    Ok(())
}
