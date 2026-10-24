#![cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};

#[test]
#[ignore = "requires MiniCPM-o AWQ checkpoint, official vision oracle and CUDA"]
fn native_siglip_resampler_matches_official_reference() -> Result<()> {
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_VISION_ORACLE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let bytes = std::fs::read(info["image"].as_str().context("missing image")?)?;
    let prepared = lia::models::minicpm_o45::prepare(&bytes, 1)?;
    ensure!(prepared.slices.len() == 1, "expected one reference slice");
    let mut engine = lia::inference::Engine::load_huggingface(&model, 128, true)?;
    let features = engine.image_embeddings(&prepared.slices[0])?;
    ensure!(
        features.len() == 64 * 4096,
        "incorrect resampler feature shape"
    );
    let reference = std::fs::read(root.join("features.f32"))?;
    ensure!(
        reference.len() == features.len() * 4,
        "truncated reference features"
    );
    let mut maximum = 0.0f32;
    let mut squared = 0.0f64;
    for (&actual, bytes) in features.iter().zip(reference.chunks_exact(4)) {
        let expected = f32::from_le_bytes(bytes.try_into().unwrap());
        ensure!(
            actual.is_finite() && expected.is_finite(),
            "non-finite image feature"
        );
        let error = actual - expected;
        maximum = maximum.max(error.abs());
        squared += f64::from(error).powi(2);
    }
    let rms = (squared / features.len() as f64).sqrt();
    println!("MiniCPM-o image features: max_abs={maximum}, rms={rms}");
    ensure!(
        maximum < 0.05 && rms < 0.01,
        "independent image feature mismatch"
    );
    Ok(())
}
