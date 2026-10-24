#![cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};
use lia::{
    models::minicpm_o45::duplex::drain::{Drain, budget},
    sampling::speech::{Parameters, Policy, Sampler},
};
use serde_json::Value;
use std::path::Path;
fn read(root: &Path, name: &Value) -> Result<Vec<f32>> {
    let bytes = std::fs::read(root.join(name.as_str().context("missing drained speech tensor")?))?;
    ensure!(
        bytes.len().is_multiple_of(4),
        "truncated drained speech tensor"
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
#[test]
#[ignore = "requires official weighted decoder/vLLM drain oracle, checkpoint and CUDA"]
fn native_sentence_and_partial_turn_drains_match_reference() -> Result<()> {
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_DRAINED_SPEECH_ORACLE")?);
    let info: Value = serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let mut engine = lia::inference::Engine::load_huggingface(&model, 64, false)?;
    engine.load_speech(&model)?;
    let empty = engine.duplex_turn_conditioning(&[], &[], true)?;
    ensure!(
        empty == read(&root, &info["empty_closed_conditioning"])? && empty.len() == 1536,
        "empty closed-sentence markers mismatch"
    );
    for group in info["groups"].as_array().context("missing drain groups")? {
        let mut cache = engine.new_speech_cache(512)?;
        let mut sampler = Sampler::new(
            Parameters {
                policy: Policy::Vllm,
                temperature: 0.8,
                top_k: 100,
                top_p: 0.8,
                repetition_penalty: 1.05,
                min_tokens: 0,
            },
            group["seed"].as_u64().context("missing drain seed")?,
        )?;
        let chunks = group["chunks"]
            .as_array()
            .context("missing drained chunks")?;
        for (index, chunk) in chunks.iter().enumerate() {
            let tokens: Vec<u32> = serde_json::from_value(chunk["tokens"].clone())?;
            let sentence = chunk["closed_sentence"] == true;
            let conditioning = engine.duplex_turn_conditioning(
                &tokens,
                &read(&root, &chunk["hidden"])?,
                sentence,
            )?;
            let expected = read(&root, &chunk["conditioning"])?;
            ensure!(
                conditioning.len() == expected.len(),
                "drained condition suffix extent mismatch"
            );
            let maximum = conditioning
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f32::max);
            ensure!(maximum < 0.002, "drained semantic/text merge mismatch");
            let (max_tokens, min_tokens) = budget(index == 0, index + 1 == chunks.len());
            ensure!(
                chunk["maximum"].as_u64() == Some(max_tokens as u64)
                    && chunk["minimum"].as_u64() == Some(min_tokens as u64),
                "weighted drain oracle budget mismatch"
            );
            sampler.set_min_tokens(min_tokens)?;
            let drain = Drain::for_turn(true, index + 1 == chunks.len(), sentence);
            ensure!(
                chunk["drain"].as_u64() == Some(drain as u64),
                "weighted drain mode mismatch"
            );
            let result = engine.speech_generate_drained(
                &mut cache,
                &conditioning,
                &mut sampler,
                max_tokens,
                drain,
            )?;
            let generated = result.generation;
            println!(
                "{} chunk {index}: conditioning_max_abs={maximum}, codes={}, sampled={}, reason={:?}, position={}",
                group["name"],
                generated.codes.len(),
                generated.sampled_tokens,
                result.reason,
                cache.position()
            );
            ensure!(
                serde_json::to_value(&generated.codes)? == chunk["codes"]
                    && chunk["sampled_tokens"].as_u64() == Some(generated.sampled_tokens as u64)
                    && chunk["reason"] == format!("{:?}", result.reason)
                    && chunk["finished"] == generated.finished
                    && chunk["position"].as_u64() == Some(cache.position() as u64),
                "weighted drained speech mismatch"
            );
        }
    }
    Ok(())
}
