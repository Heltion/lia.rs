#![cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};
use lia::models::minicpm_o45::kaldi::Features;
use std::path::Path;
fn read(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(bytes.len().is_multiple_of(4), "invalid Float32 fixture");
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn compare(name: &str, actual: &[f32], expected: &[f32]) -> Result<()> {
    ensure!(
        actual.len() == 192 && expected.len() == 192,
        "invalid speaker embedding extent"
    );
    ensure!(
        actual.iter().chain(expected).all(|x| x.is_finite()),
        "nonfinite speaker embedding"
    );
    let norm = |v: &[f32]| v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
    let a = norm(actual);
    let b = norm(expected);
    ensure!(a > 0. && b > 0., "zero speaker embedding");
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
        / 192.)
        .sqrt();
    let normalized = actual
        .iter()
        .zip(expected)
        .map(|(&x, &y)| (f64::from(x) / a - f64::from(y) / b).abs())
        .fold(0., f64::max);
    let cosine = actual
        .iter()
        .zip(expected)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum::<f64>()
        / (a * b);
    println!("{name}: max_abs={maximum}, rms={rms}, normalized_max={normalized}, cosine={cosine}");
    ensure!(
        maximum < 0.01 && rms < 0.002 && normalized < 0.0005 && cosine >= 0.999998,
        "speaker reference mismatch: {name}"
    );
    Ok(())
}
#[test]
#[ignore = "requires official CAM++ checkpoint, oracle and CUDA"]
fn speaker_matches_official_reference() -> Result<()> {
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_SPEAKER_ORACLE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let mut engine = lia::inference::Engine::load_huggingface(&model, 64, false)?;
    engine.load_speaker(&model)?;
    for case in info["cases"].as_array().context("missing speaker cases")? {
        let name = case["name"].as_str().context("missing speaker case name")?;
        let features = Features {
            frames: case["frames"].as_u64().context("missing frame count")? as usize,
            values: read(&root.join(case["input"].as_str().context("missing input")?))?,
        };
        let expected = read(&root.join(case["output"].as_str().context("missing output")?))?;
        compare(name, &engine.speaker_embedding(&features)?, &expected)?;
        if let Some(pcm) = case["pcm"].as_str() {
            compare(
                &format!("{name} native fbank"),
                &engine.speaker_embedding_pcm(&read(Path::new(pcm))?)?,
                &expected,
            )?;
        }
    }
    ensure!(
        engine
            .speaker_embedding(&Features {
                frames: 3,
                values: vec![f32::NAN; 240]
            })
            .is_err(),
        "nonfinite speaker input accepted"
    );
    ensure!(
        engine
            .speaker_embedding(&Features {
                frames: 2,
                values: vec![0.; 160]
            })
            .is_err(),
        "too short speaker input accepted"
    );
    Ok(())
}
