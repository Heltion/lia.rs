use anyhow::{Context, Result, ensure};
use std::path::Path;
fn floats(root: &Path, file: &serde_json::Value) -> Result<Vec<f32>> {
    let bytes = std::fs::read(root.join(file.as_str().context("missing resampling fixture")?))?;
    ensure!(
        bytes.len().is_multiple_of(4),
        "truncated Float32 resampling fixture"
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
#[test]
#[ignore = "requires external official torchaudio resampling oracle"]
fn rational_sinc_matches_official_cached_kernel_reference() -> Result<()> {
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_RESAMPLE_ORACLE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    for case in info["cases"]
        .as_array()
        .context("missing resampling cases")?
    {
        let resampler = lia::models::minicpm_o45::resample::Resampler::new(
            case["original"].as_u64().context("missing input rate")? as u32,
            case["target"].as_u64().context("missing output rate")? as u32,
        )?;
        let actual = resampler.process(&floats(&root, &case["input"])?)?;
        let expected = floats(&root, &case["output"])?;
        ensure!(actual.len() == expected.len(), "resampled length mismatch");
        let mut maximum = 0.0f32;
        let mut squared = 0.0f64;
        for (&a, &b) in actual.iter().zip(&expected) {
            maximum = maximum.max((a - b).abs());
            squared += f64::from(a - b).powi(2);
        }
        let rms = (squared / actual.len() as f64).sqrt();
        println!(
            "{}: samples={}, max_abs={maximum}, rms={rms}",
            case["name"],
            actual.len()
        );
        ensure!(
            maximum < 3e-6 && rms < 1e-6,
            "resampling reference mismatch"
        );
    }
    Ok(())
}
