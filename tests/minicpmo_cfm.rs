#![cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};
use std::path::Path;

fn read(root: &Path, file: &serde_json::Value) -> Result<Vec<f32>> {
    let bytes = std::fs::read(root.join(file.as_str().context("missing fixture filename")?))?;
    ensure!(bytes.len().is_multiple_of(4), "invalid Float32 fixture");
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
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
        "flow output extent mismatch"
    );
    let mut maximum = 0.0f32;
    let mut squared = 0.0f64;
    for (&a, &b) in actual.iter().zip(expected) {
        ensure!(a.is_finite() && b.is_finite(), "nonfinite flow result");
        maximum = maximum.max((a - b).abs());
        squared += f64::from(a - b).powi(2);
    }
    let rms = (squared / actual.len() as f64).sqrt();
    println!("{name}: max_abs={maximum}, rms={rms}");
    ensure!(
        maximum < max_limit && rms < rms_limit,
        "flow reference mismatch for {name}"
    );
    Ok(())
}
#[test]
#[ignore = "requires official MiniCPM-o flow checkpoint, oracle fixtures and CUDA"]
fn dit_and_euler_match_official_float32_reference() -> Result<()> {
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_CFM_ORACLE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let mut engine = lia::inference::Engine::load_huggingface(&model, 64, false)?;
    engine.load_codec_flow(&model)?;
    for case in info["cases"].as_array().context("missing flow cases")? {
        let noise = read(&root, &case["noise"])?;
        let mu = read(&root, &case["mu"])?;
        let speaker = read(&root, &case["speaker"])?;
        let condition = read(&root, &case["condition"])?;
        let name = case["name"].as_str().context("missing case name")?;
        for velocity in case["velocities"]
            .as_array()
            .context("missing velocities")?
        {
            let time = velocity["time"].as_f64().context("missing timestep")? as f32;
            let actual = engine.codec_dit_velocity(&noise, &mu, &speaker, &condition, time)?;
            compare(
                &format!("{name} t={time}"),
                &actual,
                &read(&root, &velocity["file"])?,
                0.01,
                0.002,
            )?;
        }
        let actual = engine.codec_cfm_mel(&noise, &mu, &speaker, &condition, 10)?;
        compare(
            &format!("{name} Euler"),
            &actual,
            &read(&root, &case["mel"])?,
            0.02,
            0.004,
        )?;
    }
    Ok(())
}
