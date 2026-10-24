#![cfg(feature = "cuda")]
use anyhow::Result;
use lia::{inference::Engine, tokenizer::Tokenizer};

#[test]
#[ignore = "requires LIA_TEST_GSQ_MODEL and sufficient CUDA memory"]
fn sharded_moe_prefill_and_single_steps_agree() -> Result<()> {
    let path = std::env::var("LIA_TEST_GSQ_MODEL")?;
    let tokenizer = Tokenizer::from_huggingface(&path)?;
    let tokens = tokenizer.encode(
        "A short independent check of native model execution: the capital of France is",
        false,
    )?;
    assert!(tokens.len() > 8 && tokens.len() < 64);
    let mut engine = Engine::load_huggingface(&path, 64, false)?;
    assert!(!engine.mtp_enabled());
    assert_eq!(engine.max_draft(), 0);
    engine.forward_many(&tokens, false)?;
    let expected = engine.current_token()?;
    engine.reset()?;
    for &token in &tokens {
        engine.forward_many(&[token], false)?;
    }
    assert_eq!(
        engine.current_token()?,
        expected,
        "MoE prefill differs from sequential execution"
    );
    let saved = engine.snapshot()?;
    let logits = engine.forward(expected)?;
    assert!(logits.iter().all(|value| value.is_finite()));
    let next = engine.current_token()?;
    engine.restore(&saved)?;
    engine.forward(expected)?;
    assert_eq!(
        engine.current_token()?,
        next,
        "MoE snapshot replay changed output"
    );
    println!(
        "native GSQ first token {expected}: {:?}",
        String::from_utf8(tokenizer.decode(&[expected])?)?
    );
    Ok(())
}

#[test]
#[ignore = "requires real checkpoint and independently generated vision reference outside Git"]
fn native_vision_matches_official_transformer_features() -> Result<()> {
    let path = std::env::var("LIA_TEST_GSQ_MODEL")?;
    let reference = std::path::PathBuf::from(std::env::var("LIA_TEST_VISION_REFERENCE")?);
    let processor = lia::vision::Processor::from_huggingface(&path, 256, 1024)?;
    let image = processor.prepare(&std::fs::read(reference.join("vision-reference.png"))?)?;
    let mut engine = Engine::load_huggingface(&path, 64, true)?;
    let actual = engine.image_embeddings(&image)?;
    let bytes = std::fs::read(reference.join("vision-reference.f32"))?;
    anyhow::ensure!(bytes.len().is_multiple_of(4), "invalid F32 reference");
    let expected = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect::<Vec<_>>();
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().chain(&expected).all(|v| v.is_finite()));
    let mut error = 0.0f64;
    let mut norm = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut dot = 0.0f64;
    for (&a, &b) in actual.iter().zip(&expected) {
        let (a, b) = (f64::from(a), f64::from(b));
        error += (a - b).powi(2);
        norm += b * b;
        actual_norm += a * a;
        dot += a * b;
    }
    let relative_rmse = (error / norm).sqrt();
    let cosine = dot / (norm * actual_norm).sqrt();
    println!(
        "vision elements={} relative_rmse={relative_rmse} cosine={cosine}",
        actual.len()
    );
    assert!(
        relative_rmse < 0.03,
        "vision feature error exceeds FP16 computation tolerance"
    );
    assert!(
        cosine > 0.999,
        "vision feature direction differs from official reference"
    );
    Ok(())
}

#[test]
#[ignore = "requires LIA_TEST_GSQ_MODEL and sufficient CUDA memory"]
fn decode_graph_replay_preserves_sampling_across_boundaries_and_reset() -> Result<()> {
    let path = std::env::var("LIA_TEST_GSQ_MODEL")?;
    let tokenizer = Tokenizer::from_huggingface(&path)?;
    let mut prefix = tokenizer.encode(
        "An independent check of graph replay and recurrent state.",
        false,
    )?;
    let inputs = tokenizer.encode("The quick brown fox jumps over the lazy dog. ", false)?;
    if std::env::var("LIA_ATTN_MMA_DECODE").as_deref() == Ok("1") {
        prefix = prefix.into_iter().cycle().take(490).collect();
    }
    let mut engine = Engine::load_huggingface(&path, 1024, false)?;
    engine.forward_many(&prefix, false)?;
    let mut reference = Vec::new();
    let mut sampling = lia::sampling::Sampling::new(1.0, 20, 0.95, 42)?;
    for token in inputs.iter().cycle().take(132) {
        engine.forward(*token)?;
        reference.push((
            engine.current_token()?,
            engine.current_sample(&mut sampling)?,
        ));
    }
    for pass in 0..2 {
        engine.reset()?;
        engine.forward_many(&prefix, false)?;
        let before = engine.cuda_graph_stats();
        let mut sampling = lia::sampling::Sampling::new(1.0, 20, 0.95, 42)?;
        for (index, (token, expected)) in inputs.iter().cycle().zip(&reference).enumerate() {
            if index == 60 {
                assert!(engine.speculate(248320, 1, u32::MAX).is_err());
                let snapshot = engine.snapshot()?;
                let (_, next, _, _) = engine.speculate(*token, 1, u32::MAX)?;
                assert_eq!(next, expected.0, "before snapshot restore");
                engine.restore(&snapshot)?;
            }
            let (_, next, _, _) = engine.speculate(*token, 1, u32::MAX)?;
            assert_eq!(next, expected.0, "greedy pass={pass} step={index}");
            assert_eq!(
                engine.current_sample(&mut sampling)?,
                expected.1,
                "sample pass={pass} step={index}"
            );
        }
        let after = engine.cuda_graph_stats();
        if engine.cuda_graph_enabled() {
            assert!(after[2] - before[2] < 8, "decode recaptures every token");
        }
    }
    Ok(())
}
