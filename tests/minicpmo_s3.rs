#![cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};
use lia::models::minicpm_o45::audio::Features;
use std::path::Path;
fn read(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(bytes.len().is_multiple_of(4), "invalid Float32 fixture");
    Ok(bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
fn compare(name: &str, actual: &[f32], expected: &[f32], max: f32, rms_limit: f64) -> Result<()> {
    ensure!(
        actual.len() == expected.len() && !actual.is_empty(),
        "S3 extent mismatch: {name}"
    );
    ensure!(
        actual.iter().chain(expected).all(|v| v.is_finite()),
        "nonfinite S3 result"
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
        maximum < max && rms < rms_limit,
        "S3 numerical mismatch: {name}"
    );
    Ok(())
}
#[test]
#[ignore = "requires official S3 checkpoint, FP32 oracle and CUDA"]
fn s3_matches_official_reference() -> Result<()> {
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_S3_ORACLE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let mut engine = lia::inference::Engine::load_huggingface(&model, 64, false)?;
    engine.load_s3(&model)?;
    for case in info["cases"].as_array().context("missing S3 cases")? {
        let name = case["name"].as_str().context("missing S3 case name")?;
        let features = Features {
            frames: case["frames"].as_u64().context("missing S3 frame count")? as usize,
            values: read(&root.join(case["mel"].as_str().context("missing mel")?))?,
        };
        compare(
            &format!("{name} hidden"),
            &engine.s3_embeddings(&features)?,
            &read(&root.join(case["hidden"].as_str().context("missing hidden")?))?,
            0.005,
            0.0005,
        )?;
        compare(
            &format!("{name} projected"),
            &engine.s3_projection(&features)?,
            &read(&root.join(case["projected"].as_str().context("missing projection")?))?,
            0.005,
            0.0005,
        )?;
        if let Some(pcm) = case["pcm"].as_str() {
            let extracted = lia::models::minicpm_o45::audio::Processor::for_s3().extract(
                &read(Path::new(pcm))?,
                lia::models::minicpm_o45::audio::Normalization::Dynamic,
            )?;
            compare(
                &format!("{name} native mel"),
                &extracted.values,
                &features.values,
                0.0001,
                0.00001,
            )?;
            let native_codes = engine.s3_codes(&extracted)?;
            let expected: Vec<u32> = serde_json::from_value(case["codes"].clone())?;
            ensure!(
                native_codes == expected,
                "native mel S3 code mismatch: {name}"
            );
        }
        let codes = engine.s3_codes(&features)?;
        let expected: Vec<u32> = serde_json::from_value(case["codes"].clone())?;
        println!(
            "{name}: {} codes, {} mismatches",
            codes.len(),
            codes.iter().zip(&expected).filter(|(a, b)| a != b).count()
        );
        ensure!(codes == expected, "S3 code mismatch: {name}");
    }
    ensure!(
        engine
            .s3_codes(&Features {
                frames: 1,
                values: vec![f32::NAN; 128]
            })
            .is_err(),
        "invalid S3 input accepted"
    );
    ensure!(
        engine
            .s3_codes(&Features {
                frames: 3001,
                values: vec![0.; 3001 * 128]
            })
            .is_err(),
        "oversize S3 input accepted"
    );
    Ok(())
}
