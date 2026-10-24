#![cfg(feature = "cuda")]
use anyhow::{Result, ensure};
use lia::sampling::speech::Parameters;
#[test]
#[ignore = "requires MiniCPM-o checkpoint, speech conditioning oracle and CUDA"]
fn autoregressive_codes_obey_limits_callbacks_and_seed() -> Result<()> {
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_SPEECH_ORACLE")?);
    let bytes = std::fs::read(root.join("condition-31.condition.f32"))?;
    let conditioning: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let mut engine = lia::inference::Engine::load_huggingface(&model, 64, false)?;
    engine.load_speech(&model)?;
    let mut emitted = Vec::new();
    let first =
        engine.speech_generate_with(&conditioning, Parameters::default(), 64, 42, |code| {
            emitted.push(code);
            Ok(())
        })?;
    let repeat = engine.speech_generate(&conditioning, Parameters::default(), 64, 42)?;
    ensure!(
        first.codes == emitted && first.codes == repeat.codes && first.finished == repeat.finished,
        "speech callback or seed behavior mismatch"
    );
    ensure!(
        (50..=64).contains(&first.codes.len()) && first.codes.iter().all(|&code| code < 6561),
        "invalid speech code count or control emission"
    );
    ensure!(
        first.sampled_tokens == first.codes.len() + usize::from(first.finished)
            && first.conditioning_tokens == 33
            && first.prefill_seconds > 0.0
            && first.decode_seconds > 0.0,
        "incorrect speech generation accounting"
    );
    println!(
        "speech codes: count={}, sampled={}, finished={}, prefill_seconds={}, decode_seconds={}",
        first.codes.len(),
        first.sampled_tokens,
        first.finished,
        first.prefill_seconds,
        first.decode_seconds
    );
    let mut rejected = false;
    let error = engine.speech_generate_with(&conditioning, Parameters::default(), 64, 42, |_| {
        rejected = true;
        anyhow::bail!("receiver closed")
    });
    ensure!(
        rejected && error.is_err(),
        "speech callback cancellation ignored"
    );
    Ok(())
}

fn read_floats(path: impl AsRef<std::path::Path>) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    ensure!(
        bytes.len().is_multiple_of(4),
        "truncated hidden-state fixture"
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn compare(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    limit: f32,
    rms_limit: f64,
) -> Result<()> {
    ensure!(actual.len() == expected.len(), "{label} extent mismatch");
    let mut maximum = 0.0f32;
    let mut squared = 0.0f64;
    for (&a, &b) in actual.iter().zip(expected) {
        ensure!(a.is_finite() && b.is_finite(), "nonfinite {label}");
        maximum = maximum.max((a - b).abs());
        squared += f64::from(a - b).powi(2);
    }
    let rms = (squared / actual.len() as f64).sqrt();
    println!("{label}: max_abs={maximum}, rms={rms}");
    ensure!(maximum < limit && rms < rms_limit, "{label} mismatch");
    Ok(())
}
#[test]
#[ignore = "requires real text/hidden/speech oracle, checkpoint and CUDA"]
fn real_text_hidden_alignment_conditioning_and_codes_match_reference() -> Result<()> {
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_SPEECH_TEXT_ORACLE")?);
    let hidden_root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_SPEECH_HIDDEN_ORACLE")?);
    let metadata: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(hidden_root.join("info.json"))?)?;
    let expected: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let prompt: Vec<u32> = serde_json::from_value(metadata["prompt_tokens"].clone())?;
    let tokens: Vec<u32> = serde_json::from_value(metadata["text_tokens"].clone())?;
    let hidden = read_floats(hidden_root.join("normalized-hidden.f32"))?;
    let bos = metadata["tts_bos"].as_u64().unwrap() as u32;
    let eos = metadata["tts_eos"].as_u64().unwrap() as u32;
    ensure!(
        prompt.last() == Some(&bos),
        "reference prompt does not open speech"
    );
    let mut engine = lia::inference::Engine::load_huggingface(&model, 128, false)?;
    ensure!(
        engine.current_hidden_state().is_err(),
        "uninitialized hidden state accepted"
    );
    engine.forward_many(&prompt, false)?;
    let mut hidden_checks = Vec::new();
    hidden_checks.push(compare(
        "prefill last hidden",
        &engine.current_hidden_state()?,
        &hidden[(prompt.len() - 1) * 4096..prompt.len() * 4096],
        0.01,
        0.002,
    ));
    let mut text = lia::models::minicpm_o45::speech_text::Text::new(bos, eos)?;
    text.observe(bos, &[])?;
    for (index, &token) in tokens.iter().enumerate() {
        engine.forward_many(&[token], false)?;
        let actual = engine.current_hidden_state()?;
        let row = prompt.len() + index;
        hidden_checks.push(compare(
            &format!("spoken token {index}"),
            &actual,
            &hidden[row * 4096..(row + 1) * 4096],
            0.01,
            0.002,
        ));
        text.observe(token, &actual)?;
    }
    text.observe(eos, &[])?;
    let text = text.finish()?;
    ensure!(text.tokens == tokens, "speech token bounds mismatch");
    engine.load_speech(&model)?;
    let conditioning = engine.speech_conditioning(&text.tokens, &text.hidden)?;
    compare(
        "real-text speech conditioning",
        &conditioning,
        &read_floats(root.join("conditioning.f32"))?,
        0.002,
        0.0003,
    )?;
    let generated = engine.speech_generate(&conditioning, Parameters::default(), 256, 42)?;
    let codes: Vec<u32> = serde_json::from_value(expected["codes"].clone())?;
    let matched = generated
        .codes
        .iter()
        .zip(&codes)
        .take_while(|(a, b)| a == b)
        .count();
    println!(
        "real-text codes: native={}, reference={}, matching_prefix={matched}, finished={}",
        generated.codes.len(),
        codes.len(),
        generated.finished
    );
    ensure!(
        generated.codes == codes && generated.finished == expected["finished"].as_bool().unwrap(),
        "real-text speech codes diverge from reference"
    );
    for check in hidden_checks {
        check?;
    }
    Ok(())
}
