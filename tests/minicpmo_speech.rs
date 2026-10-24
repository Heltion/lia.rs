#![cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};
use std::path::Path;

fn floats(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(bytes.len().is_multiple_of(4), "truncated FP32 reference");
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn compare(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    maximum: f32,
    rms_limit: f64,
) -> Result<()> {
    ensure!(
        !actual.is_empty() && actual.len() == expected.len(),
        "{label}: shape mismatch"
    );
    let mut max = 0.0f32;
    let mut squared = 0.0f64;
    for (&a, &b) in actual.iter().zip(expected) {
        ensure!(a.is_finite() && b.is_finite(), "{label}: non-finite output");
        max = max.max((a - b).abs());
        squared += f64::from(a - b).powi(2);
    }
    let rms = (squared / actual.len() as f64).sqrt();
    println!("{label}: max_abs={max}, rms={rms}");
    ensure!(
        max < maximum && rms < rms_limit,
        "{label}: independent speech reference mismatch"
    );
    Ok(())
}

#[test]
#[ignore = "requires MiniCPM-o AWQ checkpoint, official speech decoder oracle and CUDA"]
fn native_speech_conditioning_prefill_and_decode_match_official_reference() -> Result<()> {
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_SPEECH_ORACLE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let mut engine = lia::inference::Engine::load_huggingface(&model, 64, false)?;
    engine.load_speech(&model)?;
    for case in info["cases"].as_array().context("missing cases")? {
        let label = case["name"].as_str().context("missing name")?;
        let tokens: Vec<u32> = serde_json::from_value(case["tokens"].clone())?;
        let hidden = floats(&root.join(case["hidden"].as_str().unwrap()))?;
        let conditioning = engine.speech_conditioning(&tokens, &hidden)?;
        let expected = floats(&root.join(case["condition"].as_str().unwrap()))?;
        compare(
            &format!("{label} conditioning"),
            &conditioning,
            &expected,
            0.0003,
            0.00005,
        )?;
        let mut cache = engine.new_speech_cache(128)?;
        let logits = engine.speech_logits(&mut cache, &conditioning)?;
        let expected = floats(&root.join(case["prefill"].as_str().unwrap()))?;
        compare(&format!("{label} prefill"), &logits, &expected, 0.01, 0.002)?;
        ensure!(
            cache.position() == conditioning.len() / 768,
            "incorrect speech cache advance"
        );
        for (index, step) in case["decode"].as_array().unwrap().iter().enumerate() {
            let logits =
                engine.speech_code_logits(&mut cache, step["code"].as_u64().unwrap() as u32)?;
            let expected = floats(&root.join(step["logits"].as_str().unwrap()))?;
            compare(
                &format!("{label} decode {index}"),
                &logits,
                &expected,
                0.01,
                0.002,
            )?;
        }
        cache.reset();
        let first = conditioning.len().min(768);
        engine.speech_logits(&mut cache, &conditioning[..first])?;
        let second = conditioning.len().min(first + 7 * 768);
        let logits = engine.speech_logits(&mut cache, &conditioning[first..second])?;
        let logits = if second < conditioning.len() {
            engine.speech_logits(&mut cache, &conditioning[second..])?
        } else {
            logits
        };
        let expected = floats(&root.join(case["prefill"].as_str().unwrap()))?;
        compare(
            &format!("{label} reset/chunked prefill"),
            &logits,
            &expected,
            0.01,
            0.002,
        )?;
    }
    Ok(())
}
