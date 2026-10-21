#![cfg(feature = "cuda")]

use anyhow::{Context, Result, ensure};
use lia::inference::Engine;

fn prefill(engine: &mut Engine, tokens: &[u32]) -> Result<()> {
    let mut offset = 0;
    while offset < tokens.len() {
        let count = engine.prefill_chunk(tokens.len() - offset)?;
        engine.forward_many(&tokens[offset..offset + count], false)?;
        offset += count;
    }
    Ok(())
}

#[test]
#[ignore = "requires local MiniCPM checkpoint and external multi-turn token sequences"]
fn restored_prefix_matches_fresh_execution() -> Result<()> {
    let model = std::env::var("LIA_TEST_MODEL").context("set LIA_TEST_MODEL")?;
    let path = std::env::var("LIA_TEST_CACHE_TOKENS").context("set LIA_TEST_CACHE_TOKENS")?;
    let sequences: Vec<Vec<u32>> =
        serde_json::from_slice(&std::fs::read(&path).with_context(|| format!("reading {path}"))?)
            .context("parsing external token sequences")?;
    ensure!(sequences.len() >= 2, "need at least two token sequences");
    let first = &sequences[0];
    let last = sequences.last().unwrap();
    let common = first.iter().zip(last).take_while(|(a, b)| a == b).count();
    ensure!(
        common > 0 && common < first.len() && common < last.len(),
        "need a shared prefix and distinct continuations"
    );
    let capacity = sequences.iter().map(Vec::len).max().unwrap() + 64;
    let mut engine = Engine::load_huggingface(&model, capacity, false)?;
    prefill(&mut engine, &first[..common])?;
    let snapshot = engine.snapshot()?;
    prefill(&mut engine, &first[common..])?;
    // Exercise generated-token execution before restoring the earlier prompt.
    for &token in sequences[1].iter().skip(common).take(32) {
        engine.speculate(token, 1, u32::MAX)?;
    }
    engine.restore(&snapshot)?;
    ensure!(snapshot.position == common, "snapshot position differs");
    prefill(&mut engine, &last[common..])?;
    let restored = engine.current_logits()?;
    engine.reset()?;
    prefill(&mut engine, &last[..common])?;
    prefill(&mut engine, &last[common..])?;
    let fresh = engine.current_logits()?;
    ensure!(restored.len() == fresh.len(), "logit shape differs");
    let mut difference = 0.0f64;
    let mut norm = 0.0f64;
    let mut maximum = 0.0f64;
    for (&a, &b) in restored.iter().zip(&fresh) {
        ensure!(a.is_finite() && b.is_finite(), "nonfinite cached logits");
        let delta = f64::from(a) - f64::from(b);
        difference += delta * delta;
        norm += f64::from(b).powi(2);
        maximum = maximum.max(delta.abs());
    }
    let relative = (difference / norm).sqrt();
    eprintln!(
        "MiniCPM restored prefix={common} total={} relative_l2={relative} max_abs={maximum}",
        last.len()
    );
    ensure!(
        restored
            .iter()
            .zip(&fresh)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "cache restore differs from fresh execution with the same partition"
    );
    Ok(())
}
