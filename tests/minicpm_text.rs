#![cfg(feature = "cuda")]

use anyhow::{Context, Result, ensure};
use lia::inference::Engine;
use std::path::PathBuf;

#[test]
#[ignore = "requires a local MiniCPM checkpoint and external official FP32 reference"]
fn checkpoint_logits_match_official_fp32_reference() -> Result<()> {
    let model = std::env::var("LIA_TEST_MODEL").context("set LIA_TEST_MODEL")?;
    let directory = PathBuf::from(
        std::env::var("LIA_TEST_TEXT_REFERENCE").context("set LIA_TEST_TEXT_REFERENCE")?,
    );
    let metadata: serde_json::Value = serde_json::from_slice(
        &std::fs::read(directory.join("info.json")).context("reading text reference metadata")?,
    )
    .context("parsing text reference metadata")?;
    let tokens = metadata["tokens"]
        .as_array()
        .context("missing reference tokens")?
        .iter()
        .map(|v| {
            u32::try_from(v.as_u64().context("invalid reference token")?)
                .context("reference token overflow")
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        !tokens.is_empty() && tokens.len() <= 64,
        "invalid reference length"
    );
    let bytes = std::fs::read(directory.join("logits.f32"))
        .context("reading official text reference logits")?;
    let mut engine = Engine::load_huggingface(&model, 64, false)?;
    let vocabulary = engine.vocabulary_size();
    ensure!(
        bytes.len() == tokens.len() * vocabulary * 4,
        "reference logit shape differs"
    );
    let expected = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    let mut actual = Vec::with_capacity(expected.len());
    for token in tokens {
        actual.extend(engine.forward(token)?);
    }
    ensure!(actual.len() == expected.len(), "native logit shape differs");
    ensure!(
        actual.iter().chain(&expected).all(|x| x.is_finite()),
        "nonfinite logits"
    );
    let mut dot = 0.0f64;
    let mut error = 0.0f64;
    let mut native_norm = 0.0f64;
    let mut reference_norm = 0.0f64;
    let mut maximum = 0.0f64;
    for (&a, &b) in actual.iter().zip(&expected) {
        let (a, b) = (f64::from(a), f64::from(b));
        dot += a * b;
        error += (a - b).powi(2);
        native_norm += a * a;
        reference_norm += b * b;
        maximum = maximum.max((a - b).abs());
    }
    for (a, b) in actual
        .chunks_exact(vocabulary)
        .zip(expected.chunks_exact(vocabulary))
    {
        let best = |row: &[f32]| {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(id, _)| id)
        };
        ensure!(best(a) == best(b), "native and reference top token differ");
    }
    let cosine = dot / (native_norm * reference_norm).sqrt();
    let relative = (error / reference_norm).sqrt();
    eprintln!(
        "MiniCPM text elements={} cosine={cosine} relative_l2={relative} max_abs={maximum}",
        actual.len()
    );
    ensure!(
        cosine > 0.9999 && relative < 0.01,
        "text logits exceed mixed-precision tolerance"
    );
    Ok(())
}

#[test]
#[ignore = "diagnostic helper requiring external model, request and output paths"]
fn snapshot_real_request_logits() -> Result<()> {
    let model = std::env::var("LIA_TEST_MODEL").context("set LIA_TEST_MODEL")?;
    let request_path = std::env::var("LIA_TEST_REQUEST").context("set LIA_TEST_REQUEST")?;
    let output = std::env::var("LIA_TEST_LOGITS_PATH").context("set LIA_TEST_LOGITS_PATH")?;
    let request: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&request_path).context("reading external diagnostic request")?,
    )
    .context("parsing external diagnostic request")?;
    let template = std::fs::read_to_string(PathBuf::from(&model).join("chat_template.jinja"))
        .context("reading checkpoint template")?;
    let tokenizer = lia::tokenizer::Tokenizer::from_huggingface(&model)?;
    let text = lia::chat::render_template(&template, &request)?;
    let tokens = tokenizer.encode(&text, true)?;
    let mut sources = Vec::new();
    for message in request["messages"].as_array().context("missing messages")? {
        if let Some(parts) = message["content"].as_array() {
            for part in parts {
                if part["type"] == "image_url" {
                    sources.push(
                        part["image_url"]["url"]
                            .as_str()
                            .context("missing image URL")?
                            .to_owned(),
                    );
                }
            }
        }
    }
    let mut engine = Engine::load_huggingface(&model, 65536, !sources.is_empty())?;
    let tokens = if sources.is_empty() {
        tokens
    } else {
        let prepared = lia::models::minicpm_v47::prepare_prompt(
            &tokenizer,
            tokens,
            &sources,
            9,
            engine.capacity(),
        )?;
        if let Ok(path) = std::env::var("LIA_TEST_VISION_FEATURES_PATH") {
            let mut bytes = Vec::new();
            for (_, image) in &prepared.images {
                bytes.extend(
                    engine
                        .image_embeddings(image)?
                        .into_iter()
                        .flat_map(f32::to_le_bytes),
                );
            }
            std::fs::write(&path, bytes).context("writing external diagnostic image features")?;
        }
        engine.prepare_image_inputs(&prepared.images, prepared.positions)?;
        prepared.tokens
    };
    let scalar = std::env::var("LIA_TEST_SCALAR_PREFILL").as_deref() == Ok("1");
    let mut offset = 0;
    while offset < tokens.len() {
        let count = if scalar {
            1
        } else {
            engine.prefill_chunk(tokens.len() - offset)?
        };
        engine.forward_many(&tokens[offset..offset + count], false)?;
        offset += count;
    }
    let logits = engine.current_logits()?;
    ensure!(
        logits.len() == engine.vocabulary_size() && logits.iter().all(|x| x.is_finite()),
        "invalid diagnostic logits"
    );
    let bytes: Vec<u8> = logits.into_iter().flat_map(f32::to_le_bytes).collect();
    std::fs::write(&output, bytes).context("writing external diagnostic logits")?;
    eprintln!(
        "real request prompt_tokens={} scalar={scalar}",
        tokens.len()
    );
    Ok(())
}
