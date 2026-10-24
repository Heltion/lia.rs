//! Explicit native-session setup; unsupported negotiation is rejected.
use super::*;
use crate::{
    inference::duplex::Conversation,
    models::minicpm_o45::duplex::Control,
    sampling::{duplex, speech},
};
pub(super) fn open(
    session: &mut Session,
    tokenizer: &Tokenizer,
    path: &str,
    config: &Json,
) -> Result<Conversation> {
    anyhow::ensure!(
        config.is_object(),
        "session.update requires a session object"
    );
    for key in config
        .as_object()
        .context("missing realtime session object")?
        .keys()
    {
        anyhow::ensure!(
            [
                "instructions",
                "input_audio_format",
                "output_audio_format",
                "turn_detection",
                "tools",
                "temperature",
                "seed",
                "modalities",
                "model",
                "type",
                "extra_body"
            ]
            .contains(&key.as_str()),
            "unsupported realtime session setting: {key}"
        );
    }
    anyhow::ensure!(
        config
            .get("type")
            .is_none_or(|v| v.as_str() == Some("realtime")),
        "session type must be realtime"
    );
    if let Some(extra) = config.get("extra_body") {
        let extra = extra.as_object().context("invalid realtime extra_body")?;
        anyhow::ensure!(
            extra.keys().all(|k| k == "auto_response")
                && extra
                    .get("auto_response")
                    .is_none_or(|v| v.as_bool() == Some(true)),
            "native duplex extra_body currently supports only auto_response=true"
        );
    }
    if let Some(modalities) = config.get("modalities") {
        let values = modalities
            .as_array()
            .context("invalid realtime modalities")?;
        anyhow::ensure!(
            values.len() == 2
                && values.contains(&json!("text"))
                && values.contains(&json!("audio")),
            "native realtime currently requires text and audio output together"
        );
    }
    if let Some(model) = config.get("model") {
        anyhow::ensure!(
            matches!(model.as_str(), Some("lia.rs" | "OpenBMB/MiniCPM-o-4_5-awq")),
            "requested realtime model differs from loaded MiniCPM-o 4.5"
        );
    }
    for key in ["input_audio_format", "output_audio_format"] {
        anyhow::ensure!(
            config.get(key).is_none_or(|v| v.as_str() == Some("pcm16")),
            "native realtime currently requires pcm16 audio formats"
        );
    }
    anyhow::ensure!(
        config.get("turn_detection").is_none_or(Json::is_null),
        "native model turns require turn_detection=null; external VAD is not implemented"
    );
    anyhow::ensure!(
        config
            .get("tools")
            .is_none_or(|v| v.as_array().is_some_and(Vec::is_empty)),
        "realtime tool calls are not implemented"
    );
    let engine = &mut session.engine;
    engine.load_audio(path)?;
    engine.load_speech(path)?;
    engine.load_codec_encoder(path)?;
    engine.load_codec_flow(path)?;
    engine.load_codec_hift(path)?;
    engine.load_reference_voice_weights(path)?;
    let reference = std::fs::read(std::path::Path::new(path).join("assets/HT_ref_audio.wav"))
        .context("reading bundled duplex reference audio")?;
    let voice = engine.prepare_voice_wav(&reference, 6.)?;
    let reference_features =
        crate::models::minicpm_o45::duplex::context::reference_features(&reference)?;
    let instructions = config["instructions"]
        .as_str()
        .filter(|s| !s.is_empty())
        .unwrap_or("Streaming Omni Conversation.");
    let (prefix, suffix) = crate::models::minicpm_o45::duplex::context::system_texts(instructions);
    let prefix = tokenizer.encode(&prefix, true)?;
    let suffix = tokenizer.encode(&suffix, true)?;
    let raw: Json = serde_json::from_reader(std::fs::File::open(
        std::path::Path::new(path).join("tokenizer_config.json"),
    )?)?;
    let mut forbidden: Vec<u32> = serde_json::from_value(raw["bad_token_ids"].clone())?;
    forbidden.push(151722);
    let temperature = config["temperature"].as_f64().unwrap_or(0.7);
    anyhow::ensure!(
        temperature.is_finite() && temperature >= 0.,
        "invalid realtime temperature"
    );
    let seed = config["seed"].as_u64().unwrap_or(42);
    let conversation = Conversation::new(
        engine,
        &prefix,
        Control::new(20, 0, true)?,
        duplex::Sampler::new(
            duplex::Parameters {
                greedy: temperature == 0.,
                temperature: if temperature == 0. {
                    0.7
                } else {
                    temperature as f32
                },
                ..Default::default()
            },
            forbidden,
            Vec::new(),
            seed,
        )?,
        speech::Sampler::new(
            speech::Parameters {
                policy: speech::Policy::Vllm,
                temperature: 0.8,
                top_p: 0.8,
                top_k: 100,
                repetition_penalty: 1.05,
                min_tokens: 0,
            },
            seed,
        )?,
        voice,
        seed,
    )?;
    let embeddings = engine
        .audio_embeddings(&reference_features, 50)
        .context("encoding duplex reference system audio")?;
    engine
        .forward_embeddings(&embeddings, false)
        .context("prefilling duplex reference system audio")?;
    engine
        .forward_many(&suffix, false)
        .context("closing duplex reference system context")?;
    Ok(conversation)
}
