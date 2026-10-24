use anyhow::{Context, Result, ensure};
use lia::sampling::speech::{Parameters, Policy, Sampler};
use std::path::Path;
fn floats(root: &Path, file: &serde_json::Value) -> Result<Vec<f32>> {
    let bytes = std::fs::read(
        root.join(
            file.as_str()
                .context("missing speech sampler fixture path")?,
        ),
    )?;
    ensure!(bytes.len().is_multiple_of(4), "truncated sampler fixture");
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
#[test]
#[ignore = "requires external official checkpoint/vLLM speech sampling oracle"]
fn both_speech_policies_match_pinned_reference_probabilities() -> Result<()> {
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_SPEECH_SAMPLING_ORACLE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    for case in info["cases"].as_array().context("missing sampling cases")? {
        let p = &case["parameters"];
        let parameters = Parameters {
            temperature: p["temperature"].as_f64().context("temperature")? as f32,
            top_p: p["top_p"].as_f64().context("top_p")?,
            top_k: p["top_k"].as_u64().context("top_k")? as usize,
            repetition_penalty: p["repetition_penalty"]
                .as_f64()
                .context("repetition penalty")? as f32,
            min_tokens: p["min_tokens"].as_u64().context("min tokens")? as usize,
            policy: match p["policy"].as_str() {
                Some("checkpoint") => Policy::Checkpoint,
                Some("vllm") => Policy::Vllm,
                _ => anyhow::bail!("invalid oracle policy"),
            },
        };
        let history: Vec<u32> = serde_json::from_value(case["history"].clone())?;
        let sampler = Sampler::new(parameters, 42)?;
        let actual = sampler.probabilities(&floats(&root, &case["logits"])?, &history)?;
        let expected = floats(&root, &case["probabilities"])?;
        ensure!(
            actual.len() == expected.len(),
            "sampler probability shape mismatch"
        );
        let mut maximum = 0.0f64;
        for (&a, &b) in actual.iter().zip(&expected) {
            ensure!(
                a.is_finite() && a >= 0.0 && (b != 0.0 || a < 1e-10),
                "incorrect speech sampler support"
            );
            maximum = maximum.max((a - f64::from(b)).abs());
        }
        println!("{}: max_probability_abs={maximum}", case["name"]);
        ensure!(
            maximum < 2e-6 && (actual.iter().sum::<f64>() - 1.0).abs() < 1e-12,
            "speech sampling probability mismatch"
        );
    }
    Ok(())
}
