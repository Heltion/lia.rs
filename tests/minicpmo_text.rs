#![cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};
use lia::inference::Engine;

#[test]
#[ignore = "requires official MiniCPM-o AWQ checkpoint, independent Qwen3 oracle and CUDA"]
fn native_qwen3_logits_and_generation() -> Result<()> {
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let oracle = std::path::PathBuf::from(std::env::var("LIA_QWEN3_ORACLE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(oracle.join("info.json"))?)?;
    let tokens = info["tokens"]
        .as_array()
        .context("missing reference input")?
        .iter()
        .map(|v| Ok(u32::try_from(v.as_u64().context("invalid input token")?)?))
        .collect::<Result<Vec<_>>>()?;
    ensure!(tokens.len() > 1, "reference input too short");
    let bytes = std::fs::read(oracle.join("last-logits.f32"))?;
    ensure!(bytes.len().is_multiple_of(4), "truncated reference logits");
    let expected: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let mut engine = Engine::load(&model, 128)?;
    engine.forward_many(&tokens[..tokens.len() - 1], false)?;
    let actual = engine.forward(*tokens.last().unwrap())?;
    compare(&actual, &expected, 0.025, "Qwen3 independent reference")?;
    let maximum = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap()
    };
    ensure!(
        maximum(&actual) == maximum(&expected),
        "reference greedy token mismatch"
    );
    engine.reset()?;
    let prefill = engine.forward_many(&tokens, true)?;
    let last = &prefill[prefill.len() - engine.vocabulary_size()..];
    compare(last, &actual, 0.025, "Qwen3 prefill versus final decode")?;
    ensure!(
        maximum(last) == maximum(&expected),
        "prefill greedy token mismatch"
    );
    let tokenizer = lia::tokenizer::Tokenizer::from_huggingface(&model)?;
    let mut next = maximum(last) as u32;
    let mut output = Vec::new();
    for _ in 0..16 {
        if [151643, 151645].contains(&next) {
            break;
        }
        output.push(next);
        next = maximum(&engine.forward(next)?) as u32;
    }
    let text = String::from_utf8(tokenizer.decode(&output)?)?;
    println!("native Qwen3 generated {text:?}");
    ensure!(
        text.trim() == "42",
        "arithmetic generation mismatch: {text:?}"
    );
    Ok(())
}

#[test]
#[ignore = "requires official MiniCPM-o AWQ checkpoint, 34-token reference and CUDA"]
fn native_qwen3_medium_prefill_matches_reference() -> Result<()> {
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let oracle = std::path::PathBuf::from(std::env::var("LIA_QWEN3_MEDIUM_ORACLE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(oracle.join("info.json"))?)?;
    let tokens = info["tokens"]
        .as_array()
        .context("missing medium reference input")?
        .iter()
        .map(|v| Ok(u32::try_from(v.as_u64().context("invalid input token")?)?))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        tokens.len() == 34,
        "expected actual 34-token benchmark request"
    );
    let bytes = std::fs::read(oracle.join("last-logits.f32"))?;
    ensure!(bytes.len().is_multiple_of(4), "truncated reference logits");
    let expected: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let maximum = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
    };
    let mut engine = Engine::load(&model, 128)?;
    engine.forward_many(&tokens[..33], false)?;
    let actual = engine.forward(tokens[33])?;
    compare(
        &actual,
        &expected,
        0.025,
        "33-token prefill independent reference",
    )?;
    ensure!(
        maximum(&actual) == maximum(&expected),
        "reference greedy token mismatch"
    );
    engine.reset()?;
    let prefill = engine.forward_many(&tokens, true)?;
    let last = &prefill[prefill.len() - engine.vocabulary_size()..];
    compare(
        last,
        &expected,
        0.025,
        "34-token prefill independent reference",
    )?;
    ensure!(
        maximum(last) == maximum(&expected),
        "prefill greedy token mismatch"
    );
    Ok(())
}

fn compare(actual: &[f32], expected: &[f32], limit: f32, label: &str) -> Result<()> {
    ensure!(actual.len() == expected.len(), "{label}: length mismatch");
    let mut maximum = 0.0f32;
    for (&a, &b) in actual.iter().zip(expected) {
        ensure!(a.is_finite() && b.is_finite(), "{label}: non-finite logits");
        maximum = maximum.max((a - b).abs());
    }
    println!("{label}: maximum absolute error {maximum}");
    ensure!(maximum <= limit, "{label}: {maximum} exceeds {limit}");
    Ok(())
}
