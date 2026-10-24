#![cfg(feature = "cuda")]
use anyhow::{Result, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};
#[test]
#[ignore = "requires native/reference speech WAV fixtures, MiniCPM-o checkpoint and CUDA"]
fn rendered_sentence_is_recognized_without_transcript_hint() -> Result<()> {
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_VOICE_ORACLE")?);
    let tokenizer = lia::tokenizer::Tokenizer::from_huggingface(&model)?;
    let template = lia::chat::huggingface_template(&model)?;
    let request = serde_json::json!({"messages":[{"role":"user","content":"<|audio|>\nTranscribe the audio exactly."}],"chat_template_kwargs":{"enable_thinking":false,"use_tts_template":true}});
    let rendered = lia::chat::render_template(&template, &request)?;
    let prompt = tokenizer.encode(&rendered, true)?;
    let mut engine = lia::inference::Engine::load_huggingface(&model, 512, false)?;
    engine.load_audio(&model)?;
    let mut failures = Vec::new();
    for name in [
        "reference-prompt-reference",
        "reference-prompt-native",
        "no-prompt-reference",
        "no-prompt-native",
    ] {
        engine.reset()?;
        let bytes = std::fs::read(root.join(format!("{name}.wav")))?;
        let source = format!("data:audio/wav;base64,{}", STANDARD.encode(bytes));
        let prepared = lia::models::minicpm_o45::prepare_media_prompt(
            &tokenizer,
            prompt.clone(),
            &[],
            &[source],
            1,
            512,
        )?;
        engine.prepare_media_inputs(&prepared.images, &prepared.audios, prepared.positions)?;
        engine.forward_many(&prepared.tokens, false)?;
        let mut next = engine.current_token()?;
        let mut output = Vec::new();
        for _ in 0..64 {
            if [151643, 151645].contains(&next) {
                break;
            }
            output.push(next);
            next = engine
                .forward(next)?
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
        }
        let raw = String::from_utf8(tokenizer.decode_text(&output)?)?;
        let parsed = lia::chat::parse_with_thinking(&raw, &serde_json::json!([]), false)?;
        let text = parsed["content"].as_str().unwrap_or("");
        let words: Vec<String> = text
            .split(|c: char| !c.is_ascii_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(str::to_ascii_lowercase)
            .collect();
        println!("ASR {name}: {text:?}");
        if words
            != [
                "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog",
            ]
        {
            failures.push(name);
        }
    }
    ensure!(
        failures.is_empty(),
        "speech transcription mismatches: {failures:?}"
    );
    Ok(())
}
