use anyhow::{Context, Result, ensure};
use lia::sampling::duplex::{Parameters, Sampler};
use serde_json::Value;
use std::path::Path;
fn read(root: &Path, path: &Value) -> Result<Vec<f32>> {
    let bytes = std::fs::read(root.join(path.as_str().context("missing probability fixture")?))?;
    ensure!(
        bytes.len().is_multiple_of(4),
        "truncated probability tensor"
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
#[test]
#[ignore = "requires official external StreamDecoder sampling fixtures"]
fn two_stage_duplex_sampling_matches_checkpoint_probabilities_and_draws() -> Result<()> {
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_DUPLEX_SAMPLING_ORACLE")?);
    let info: Value = serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let forbidden: Vec<u32> = serde_json::from_value(info["forbidden"].clone())?;
    for case in info["cases"].as_array().context("missing sampling cases")? {
        let p = &case["parameters"];
        let parameters = Parameters {
            greedy: p["mode"] == "greedy",
            temperature: p["temperature"].as_f64().context("missing temperature")? as f32,
            top_k: p["top_k"].as_u64().context("missing top-k")? as usize,
            top_p: p["top_p"].as_f64().context("missing top-p")?,
            repetition_penalty: p["text_repetition_penalty"]
                .as_f64()
                .context("missing penalty")? as f32,
            repetition_window: p["text_repetition_window_size"]
                .as_u64()
                .context("missing window")? as usize,
            listen_id: info["listen_id"].as_u64().context("missing listen ID")? as u32,
            listen_scale: p["listen_prob_scale"]
                .as_f64()
                .context("missing listen scale")? as f32,
            listen_top_k: p["listen_top_k"].as_u64().map(|v| v as usize),
        };
        let mut sampler = Sampler::new(parameters, forbidden.clone(), Vec::new(), 42)?;
        sampler.set_history(serde_json::from_value(case["history"].clone())?)?;
        let mut draws = 0;
        let mut max = 0.0f64;
        let mut l1 = 0.0f64;
        let selected = sampler.sample_with(&read(&root, &case["logits"])?, |actual| {
            let expected = read(&root, &case["probabilities"][draws])?;
            ensure!(
                actual.len() == expected.len(),
                "probability extent mismatch"
            );
            let mut sum = 0.;
            for (&a, &b) in actual.iter().zip(&expected) {
                let error = (a - f64::from(b)).abs();
                max = max.max(error);
                sum += error;
            }
            l1 = l1.max(sum);
            let draw = case["draws"][draws]
                .as_f64()
                .context("missing controlled draw")?;
            draws += 1;
            Ok(draw)
        })?;
        ensure!(
            draws
                == case["probabilities"]
                    .as_array()
                    .context("missing distributions")?
                    .len()
                && u64::from(selected)
                    == case["selected"].as_u64().context("missing selected ID")?
                && serde_json::to_value(sampler.history())? == case["history_after"]
                && max < 1e-7
                && l1 < 1e-5,
            "duplex sampling mismatch in {}: selected={selected}, max={max}, l1={l1}",
            case["name"]
        );
        println!(
            "{}: draws={draws}, selected={selected}, probability_max={max}, l1={l1}",
            case["name"]
        );
    }
    Ok(())
}
