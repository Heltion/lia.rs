use anyhow::{Context, Result, ensure};
use std::path::Path;
fn floats(root: &Path, file: &serde_json::Value) -> Result<Vec<f32>> {
    let bytes = std::fs::read(root.join(file.as_str().context("missing fbank fixture filename")?))?;
    ensure!(
        bytes.len().is_multiple_of(4),
        "truncated fbank Float32 fixture"
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
#[test]
#[ignore = "requires external official Kaldi fbank reference fixtures"]
fn speaker_fbank_matches_official_kaldi_configuration() -> Result<()> {
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_KALDI_ORACLE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let processor = lia::models::minicpm_o45::kaldi::Processor::default();
    let mut failed = Vec::new();
    for case in info["cases"].as_array().context("missing fbank cases")? {
        let samples = floats(&root, &case["input"])?;
        for (stage, features) in [
            ("raw", processor.extract(&samples)?),
            ("centered", processor.centered(&samples)?),
        ] {
            let expected = floats(&root, &case[stage])?;
            ensure!(
                features.values.len() == expected.len()
                    && features.frames == case["frames"].as_u64().unwrap() as usize,
                "fbank frame/shape mismatch"
            );
            let mut maximum = 0.0f32;
            let mut squared = 0.0f64;
            for (&a, &b) in features.values.iter().zip(&expected) {
                ensure!(a.is_finite() && b.is_finite(), "nonfinite fbank result");
                maximum = maximum.max((a - b).abs());
                squared += f64::from(a - b).powi(2);
            }
            let rms = (squared / expected.len() as f64).sqrt();
            println!(
                "{} {stage}: frames={}, max_abs={maximum}, rms={rms}",
                case["name"], features.frames
            );
            let name = case["name"].as_str().unwrap();
            std::fs::write(
                root.join(format!("{name}-{stage}.native.f32")),
                features
                    .values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            if !(maximum < 0.0005 && rms < 0.00005) {
                failed.push(format!("{name} {stage}"));
            }
        }
    }
    ensure!(
        failed.is_empty(),
        "Kaldi fbank reference mismatches: {failed:?}"
    );
    Ok(())
}
