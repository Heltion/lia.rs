#![cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};
use std::path::Path;
fn read(root: &Path, file: &serde_json::Value) -> Result<Vec<f32>> {
    let bytes = std::fs::read(root.join(file.as_str().context("missing fixture path")?))?;
    ensure!(bytes.len().is_multiple_of(4), "invalid Float32 fixture");
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
#[test]
#[ignore = "requires official HiFT weights, F0 oracle and CUDA"]
fn f0_predictor_matches_official_float32_reference() -> Result<()> {
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_HIFT_F0_ORACLE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let mut engine = lia::inference::Engine::load_huggingface(&model, 64, false)?;
    engine.load_codec_f0(&model)?;
    for case in info["cases"].as_array().context("missing F0 cases")? {
        let mel = read(&root, &case["mel"])?;
        let expected = read(&root, &case["f0"])?;
        let actual = engine.codec_f0(&mel)?;
        ensure!(
            actual.len() == expected.len() && actual.len() * 80 == mel.len(),
            "F0 output extent mismatch"
        );
        let mut maximum = 0.0f32;
        let mut squared = 0.0f64;
        for (&a, &b) in actual.iter().zip(&expected) {
            ensure!(
                a.is_finite() && b.is_finite() && a >= 0.0,
                "invalid F0 output"
            );
            maximum = maximum.max((a - b).abs());
            squared += f64::from(a - b).powi(2);
        }
        let rms = (squared / actual.len() as f64).sqrt();
        println!("{}: max_abs={maximum}, rms={rms}", case["name"]);
        ensure!(
            maximum < 0.02 && rms < 0.004,
            "F0 predictor reference mismatch"
        );
    }
    ensure!(
        engine.codec_f0(&[]).is_err() && engine.codec_f0(&[f32::NAN; 80]).is_err(),
        "invalid F0 input accepted"
    );
    Ok(())
}

fn compare(
    name: &str,
    actual: &[f32],
    expected: &[f32],
    max_limit: f32,
    rms_limit: f64,
) -> Result<()> {
    ensure!(
        actual.len() == expected.len(),
        "HiFT output extent mismatch for {name}"
    );
    let mut maximum = 0.0f32;
    let mut squared = 0.0f64;
    for (&a, &b) in actual.iter().zip(expected) {
        ensure!(
            a.is_finite() && b.is_finite(),
            "nonfinite HiFT result for {name}"
        );
        maximum = maximum.max((a - b).abs());
        squared += f64::from(a - b).powi(2);
    }
    let rms = (squared / actual.len() as f64).sqrt();
    println!("{name}: max_abs={maximum}, rms={rms}");
    ensure!(
        maximum < max_limit && rms < rms_limit,
        "HiFT reference mismatch for {name}"
    );
    Ok(())
}
#[test]
#[ignore = "requires official HiFT decoder oracle, checkpoint and CUDA"]
fn mel_source_decoder_matches_official_float32_reference() -> Result<()> {
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_HIFT_DECODER_ORACLE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let mut engine = lia::inference::Engine::load_huggingface(&model, 64, false)?;
    engine.load_codec_hift(&model)?;
    for case in info["cases"]
        .as_array()
        .context("missing HiFT decoder cases")?
    {
        let mel = read(&root, &case["mel"]["file"])?;
        let source = read(&root, &case["source"]["file"])?;
        let spectrum = engine.codec_hift_spectrum(&mel, &source)?;
        let name = case["name"].as_str().context("missing HiFT case name")?;
        compare(
            &format!("{name} spectrum"),
            &spectrum,
            &read(&root, &case["intermediates"]["conv_post"]["file"])?,
            0.03,
            0.005,
        )?;
        let wave = engine.codec_hift_waveform(&mel, &source)?;
        ensure!(
            wave.len() == source.len() && wave.iter().all(|v| v.abs() <= 0.99),
            "invalid synthesized waveform extent or range"
        );
        compare(
            &format!("{name} waveform"),
            &wave,
            &read(&root, &case["wave"]["file"])?,
            0.01,
            0.002,
        )?;
    }
    ensure!(
        engine.codec_hift_waveform(&[], &[]).is_err()
            && engine.codec_hift_waveform(&[0.; 80], &[0.; 479]).is_err(),
        "invalid HiFT input accepted"
    );
    Ok(())
}

#[test]
#[ignore = "requires official HiFT source oracle, checkpoint and CUDA"]
fn harmonic_source_matches_official_fixed_noise_reference() -> Result<()> {
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_HIFT_SOURCE_ORACLE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let mut engine = lia::inference::Engine::load_huggingface(&model, 64, false)?;
    engine.load_codec_hift(&model)?;
    for case in info["cases"]
        .as_array()
        .context("missing HiFT source cases")?
    {
        let f0 = read(&root, &case["f0"])?;
        let noise = read(&root, &case["noise"])?;
        let actual = engine.codec_harmonic_source(&f0, &noise)?;
        compare(
            case["name"].as_str().context("missing source name")?,
            &actual,
            &read(&root, &case["source"])?,
            0.003,
            0.0003,
        )?;
    }
    Ok(())
}

#[test]
#[ignore = "requires real speech mel fixtures, HiFT checkpoint and CUDA"]
fn seeded_mel_synthesis_is_finite_reproducible_and_nonconstant() -> Result<()> {
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_HIFT_F0_ORACLE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let case = info["cases"]
        .as_array()
        .context("missing mel cases")?
        .iter()
        .find(|c| c["frames"] == 33)
        .context("missing 33-frame mel")?;
    let mel = read(&root, &case["mel"])?;
    let mut engine = lia::inference::Engine::load_huggingface(&model, 64, false)?;
    engine.load_codec_hift(&model)?;
    let a = engine.codec_synthesize(&mel, 10427)?;
    let b = engine.codec_synthesize(&mel, 10427)?;
    let c = engine.codec_synthesize(&mel, 10428)?;
    ensure!(
        a.len() == 33 * 480 && a.iter().all(|v| v.is_finite() && v.abs() <= 0.99),
        "invalid synthesized audio"
    );
    ensure!(a == b && a != c, "synthesis seed behavior mismatch");
    let rms = (a.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / a.len() as f64).sqrt();
    println!("seeded mel synthesis: samples={}, rms={rms}", a.len());
    ensure!(rms > 0.0001, "synthesis produced effectively silent audio");
    Ok(())
}
