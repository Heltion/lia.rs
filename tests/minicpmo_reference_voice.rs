#![cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};
use std::path::Path;
fn read(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(bytes.len().is_multiple_of(4), "invalid Float32 fixture");
    Ok(bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
fn compare(name: &str, actual: &[f32], expected: &[f32], limit: f32, rms_limit: f64) -> Result<()> {
    ensure!(
        actual.len() == expected.len() && !actual.is_empty(),
        "reference voice extent mismatch: {name}"
    );
    ensure!(
        actual.iter().chain(expected).all(|v| v.is_finite()),
        "nonfinite reference voice value"
    );
    let maximum = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max);
    let rms = (actual
        .iter()
        .zip(expected)
        .map(|(a, b)| f64::from(a - b).powi(2))
        .sum::<f64>()
        / actual.len() as f64)
        .sqrt();
    println!("{name}: max_abs={maximum}, rms={rms}");
    ensure!(
        maximum < limit && rms < rms_limit,
        "reference voice mismatch: {name}"
    );
    Ok(())
}
#[test]
#[ignore = "requires official reference voice fixtures, checkpoint and CUDA"]
fn waveform_to_voice_matches_official_pipeline() -> Result<()> {
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_REFERENCE_VOICE_ORACLE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let mut engine = lia::inference::Engine::load_huggingface(&model, 64, false)?;
    engine.load_reference_voice_weights(&model)?;
    for case in info["cases"]
        .as_array()
        .context("missing reference voice cases")?
    {
        let name = case["name"].as_str().context("missing case name")?;
        let bytes = std::fs::read(case["input"].as_str().context("missing input WAV")?)?;
        let seconds = case["seconds"]
            .as_f64()
            .context("missing reference window")?;
        let normalized = lia::models::minicpm_o45::reference::normalize_wav(&bytes, seconds)?;
        // Tiny resampling differences can cross a PCM16 floor boundary by one LSB.
        compare(
            &format!("{name} normalized"),
            &normalized,
            &read(
                &root.join(
                    case["normalized"]
                        .as_str()
                        .context("missing normalized PCM")?,
                ),
            )?,
            1. / 32768. + 2.4e-7,
            0.00001,
        )?;
        let start = std::time::Instant::now();
        let prepared = engine.prepare_voice_wav(&bytes, seconds)?;
        let voice = prepared.voice()?;
        let codes: Vec<u32> = serde_json::from_value(case["codes"].clone())?;
        println!(
            "{name}: {} codes, {} mismatches, preparation_seconds={}",
            voice.codes().len(),
            voice
                .codes()
                .iter()
                .zip(&codes)
                .filter(|(a, b)| a != b)
                .count(),
            start.elapsed().as_secs_f64()
        );
        ensure!(
            voice.codes() == codes,
            "reference voice code mismatch: {name}"
        );
        compare(
            &format!("{name} speaker"),
            voice.embedding(),
            &read(&root.join(case["speaker"].as_str().context("missing speaker")?))?,
            0.01,
            0.002,
        )?;
        compare(
            &format!("{name} mel"),
            voice.mel(),
            &read(&root.join(case["mel"].as_str().context("missing mel")?))?,
            0.0005,
            0.00005,
        )?;
    }
    ensure!(
        engine.prepare_voice_wav(&[], 6.).is_err(),
        "empty reference accepted"
    );
    Ok(())
}
