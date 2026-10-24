#![cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};
use std::path::Path;
fn read(root: &Path, file: &serde_json::Value) -> Result<Vec<f32>> {
    let bytes = std::fs::read(root.join(file.as_str().context("missing HiFT stream fixture")?))?;
    ensure!(
        bytes.len().is_multiple_of(4),
        "invalid Float32 HiFT fixture"
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
#[test]
#[ignore = "requires official HiFT streaming fixtures, checkpoint and CUDA"]
fn source_and_wave_overlaps_match_official_reference() -> Result<()> {
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_CODEC_HIFT_STREAM_ORACLE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let mut engine = lia::inference::Engine::load_huggingface(&model, 64, false)?;
    engine.load_codec_hift(&model)?;
    let mut cache = engine.new_codec_hift_cache()?;
    let mut combined = Vec::new();
    let mut expected_total = 0;
    ensure!(
        cache.position() == 0 && cache.history_frames() == 0 && !cache.finished(),
        "invalid initial HiFT cache"
    );
    ensure!(
        engine
            .codec_hift_chunk(&mut cache, &[0.; 80], &vec![0.; 480 * 9], false)
            .is_err(),
        "nonfinal chunk without sufficient overlap accepted"
    );
    ensure!(
        cache.position() == 0 && cache.history_frames() == 0,
        "invalid HiFT input advanced cache"
    );
    for (index, chunk) in info["chunks"]
        .as_array()
        .context("missing HiFT chunks")?
        .iter()
        .enumerate()
    {
        let mel = read(&root, &chunk["mel"])?;
        let noise = read(&root, &chunk["noise"])?;
        let last = chunk["last"].as_bool().context("missing final flag")?;
        let actual = engine.codec_hift_chunk(&mut cache, &mel, &noise, last)?;
        let expected = read(&root, &chunk["wave"])?;
        ensure!(
            actual.len() == expected.len()
                && actual.len()
                    == chunk["samples"].as_u64().context("missing sample count")? as usize,
            "HiFT emitted sample count mismatch"
        );
        ensure!(
            actual.iter().chain(&expected).all(|v| v.is_finite()),
            "nonfinite HiFT stream waveform"
        );
        let mut error = 0.;
        let mut energy = 0.;
        let mut actual_energy = 0.;
        let mut dot = 0.;
        for (&a, &b) in actual.iter().zip(&expected) {
            let (a, b) = (f64::from(a), f64::from(b));
            error += (a - b).powi(2);
            energy += b * b;
            actual_energy += a * a;
            dot += a * b;
        }
        ensure!(
            energy > 0. && actual_energy > 0.,
            "silent HiFT stream fixture/result"
        );
        let relative = (error / energy).sqrt();
        let cosine = dot / (energy * actual_energy).sqrt();
        println!(
            "chunk {index}: samples={}, relative_l2={relative}, cosine={cosine}, position={}",
            actual.len(),
            cache.position()
        );
        ensure!(
            relative < 0.5 && cosine > 0.9,
            "HiFT overlap reference mismatch"
        );
        expected_total += expected.len();
        combined.extend(actual);
        ensure!(
            cache.position() == expected_total
                && cache.finished() == last
                && cache.history_frames() == 8,
            "HiFT stream lifecycle mismatch"
        );
    }
    ensure!(
        engine
            .codec_hift_chunk(&mut cache, &[0.; 80], &[], true)
            .is_err(),
        "finished HiFT cache reused"
    );
    ensure!(
        combined.len() == 75840,
        "complete HiFT stream has missing or duplicated samples"
    );
    std::fs::write(
        root.join("native-stream.wav"),
        lia::models::minicpm_o45::wav::pcm16(&combined, 24000)?,
    )?;
    let first = info["chunks"][0].clone();
    let mel = read(&root, &first["mel"])?;
    let noise = read(&root, &first["noise"])?;
    let mut tiny = engine.new_codec_hift_cache()?;
    let wave = engine.codec_hift_chunk(&mut tiny, &mel[..80], &noise[..480 * 9], true)?;
    ensure!(
        wave.len() == 480
            && wave.iter().all(|v| v.is_finite())
            && tiny.finished()
            && tiny.position() == 480,
        "one-frame final HiFT chunk failed"
    );
    Ok(())
}
