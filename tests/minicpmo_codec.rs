#![cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};

#[test]
#[ignore = "requires MiniCPM-o checkpoint, official codec encoder oracle and CUDA"]
fn upsample_encoder_matches_official_float32_reference() -> Result<()> {
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_CODEC_ENCODER_ORACLE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let mut engine = lia::inference::Engine::load_huggingface(&model, 64, false)?;
    engine.load_codec_encoder(&model)?;
    for case in info["cases"].as_array().context("missing codec cases")? {
        let codes: Vec<u32> = serde_json::from_value(case["codes"].clone())?;
        let actual = engine.codec_encoder_features(&codes)?;
        let bytes = std::fs::read(root.join(case["features"].as_str().unwrap()))?;
        ensure!(
            actual.len() == codes.len() * 2 * 80 && bytes.len() == actual.len() * 4,
            "codec output shape mismatch"
        );
        let mut maximum = 0.0f32;
        let mut squared = 0.0f64;
        for (&a, b) in actual.iter().zip(bytes.chunks_exact(4)) {
            let expected = f32::from_le_bytes(b.try_into().unwrap());
            ensure!(
                a.is_finite() && expected.is_finite(),
                "nonfinite codec feature"
            );
            maximum = maximum.max((a - expected).abs());
            squared += f64::from(a - expected).powi(2);
        }
        let rms = (squared / actual.len() as f64).sqrt();
        println!("{}: max_abs={maximum}, rms={rms}", case["name"]);
        ensure!(
            maximum < 0.01 && rms < 0.002,
            "independent upsample encoder mismatch"
        );
    }
    Ok(())
}
