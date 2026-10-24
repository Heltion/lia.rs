#![cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};
use lia::{
    inference::{AudioCache, Engine},
    models::minicpm_o45::{
        audio::stream::Streaming,
        duplex::{Control, LISTEN},
    },
    sampling::duplex::{Parameters, Sampler},
};
use serde_json::Value;
use std::{collections::VecDeque, path::Path};
fn read(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(
        bytes.len().is_multiple_of(4),
        "truncated duplex Float32 fixture"
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
fn input(
    engine: &Engine,
    audio: &mut AudioCache,
    processor: &mut Streaming,
    unit: &Value,
    native: bool,
) -> Result<Vec<f32>> {
    if native {
        let samples = read(Path::new(
            unit["pcm"].as_str().context("missing duplex PCM")?,
        ))?;
        let features = processor.push(&samples, false)?.features;
        engine.audio_stream_embeddings(
            audio,
            &features,
            unit["prefix_frames"].as_u64().context("missing prefix")? as usize,
            2,
        )
    } else {
        read(Path::new(
            unit["input"].as_str().context("missing input embeddings")?,
        ))
    }
}
fn error(actual: &[f32], expected: &[f32]) -> Result<f32> {
    ensure!(
        actual.len() == expected.len() && actual.iter().chain(expected).all(|v| v.is_finite()),
        "duplex output shape/value mismatch"
    );
    Ok(actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max))
}
#[test]
#[ignore = "requires official Qwen3 duplex-unit oracle, checkpoint and CUDA"]
fn native_units_preserve_language_history_and_own_token_hidden_rows() -> Result<()> {
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_DUPLEX_UNIT_ORACLE")?);
    let info: Value = serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let prefix: Vec<u32> = serde_json::from_value(info["prefix_tokens"].clone())?;
    let config: Value = serde_json::from_reader(std::fs::File::open(
        Path::new(&model).join("tokenizer_config.json"),
    )?)?;
    let mut forbidden: Vec<u32> = serde_json::from_value(config["bad_token_ids"].clone())?;
    forbidden.push(151722);
    let mut engine = Engine::load_huggingface(&model, 128, false)?;
    engine.load_audio(&model)?;
    for native in [false, true] {
        engine.reset()?;
        engine.forward_many(&prefix, false)?;
        let mut control = Control::new(20, 0, true)?;
        ensure!(
            engine
                .duplex_unit(&mut control, &[f32::NAN; 4096], |_| Ok(LISTEN))
                .is_err()
                && !control.active()
                && engine.position() == prefix.len(),
            "invalid unit input changed state"
        );
        let mut audio = engine.new_audio_cache()?;
        let mut processor = Streaming::new(1000, 1035, 20, false)?;
        let mut history = Vec::new();
        for (index, unit) in info["units"]
            .as_array()
            .context("missing units")?
            .iter()
            .enumerate()
        {
            let values = input(&engine, &mut audio, &mut processor, unit, native)?;
            let mut samples: VecDeque<u32> = serde_json::from_value(unit["samples"].clone())?;
            history.extend(samples.iter().copied().filter(|&v| v != 151718));
            let output = engine.duplex_unit(&mut control, &values, |logits| {
                ensure!(
                    logits.len() == 151748 && logits.iter().all(|v| v.is_finite()),
                    "invalid unit sampling logits"
                );
                samples
                    .pop_front()
                    .context("unit exhausted controlled decisions")
            })?;
            ensure!(
                samples.is_empty()
                    && serde_json::to_value(&output.decision_tokens)? == unit["decisions"]
                    && serde_json::to_value(&output.condition.tokens)? == unit["hidden_tokens"]
                    && unit["is_listen"] == output.is_listen
                    && unit["end_of_turn"] == output.end_of_turn
                    && output.start_position
                        == unit["start_position"]
                            .as_u64()
                            .context("missing unit start")? as usize
                    && output.end_position
                        == unit["end_position"].as_u64().context("missing unit end")? as usize,
                "unit protocol/position mismatch"
            );
            let expected =
                read(&root.join(unit["hidden"].as_str().context("missing hidden rows")?))?;
            let maximum = error(&output.condition.hidden, &expected)?;
            println!(
                "unit {index} native_pcm={native}: hidden_max_abs={maximum}, rows={}, positions={}..{}",
                output.condition.tokens.len(),
                output.start_position,
                output.end_position
            );
            ensure!(maximum < 0.03, "duplex hidden conditioning mismatch");
        }
        let values = input(
            &engine,
            &mut audio,
            &mut processor,
            &serde_json::json!({"input":info["probe_input"],"pcm":info["probe_pcm"],"prefix_frames":2}),
            native,
        )?;
        let expected = read(&root.join("last-logits.f32"))?;
        ensure!(
            info["top_ids"][0].as_u64() == Some(u64::from(LISTEN)),
            "probe fixture is not a native listen decision"
        );
        let mut sampler = Sampler::new(
            Parameters {
                greedy: true,
                ..Parameters::default()
            },
            forbidden.clone(),
            Vec::new(),
            42,
        )?;
        sampler.set_history(history)?;
        let mut calls = 0;
        let mut maximum = 0.0f32;
        let output = engine.duplex_unit(&mut control, &values, |logits| {
            calls += 1;
            maximum = error(logits, &expected)?;
            sampler.sample(logits)
        })?;
        println!(
            "native sampled unit native_pcm={native}: logits_max_abs={maximum}, calls={calls}, tokens={:?}",
            output.decision_tokens
        );
        ensure!(
            maximum < 0.025
                && calls == 1
                && output.is_listen
                && output.decision_tokens == [LISTEN]
                && output.condition.tokens.is_empty()
                && output.condition.hidden.is_empty()
                && control.turn_ended(),
            "native sampled duplex unit mismatch"
        );
    }
    Ok(())
}
