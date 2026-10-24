//! Explicit audio modalities and checkpoint template selection.
use super::*;
use crate::models::minicpm_o45::speech_sampling;
use base64::{Engine as _, engine::general_purpose::STANDARD};
pub(super) struct Options {
    pub text: bool,
    pub format: String,
    pub maximum: usize,
    pub seed: u64,
    pub seconds: f64,
    pub reference: Option<Vec<u8>>,
    pub parameters: crate::sampling::speech::Parameters,
}
pub(crate) fn requested(request: &Json) -> Result<bool> {
    let Some(value) = request.get("modalities").filter(|v| !v.is_null()) else {
        return Ok(false);
    };
    let values = value.as_array().context("modalities must be an array")?;
    anyhow::ensure!(
        !values.is_empty() && values.iter().all(|v| v == "text" || v == "audio"),
        "unsupported output modality"
    );
    Ok(values.iter().any(|v| v == "audio"))
}
pub(in crate::server) fn prepare<'a>(
    input: &'a Json,
    supported: bool,
) -> Result<std::borrow::Cow<'a, Json>> {
    if !requested(input)? {
        return Ok(std::borrow::Cow::Borrowed(input));
    }
    anyhow::ensure!(supported, "this model does not support speech output");
    let mut input = input.clone();
    if input.get("chat_template_kwargs").is_none_or(Json::is_null) {
        input["chat_template_kwargs"] = json!({});
    }
    anyhow::ensure!(
        input["chat_template_kwargs"].is_object(),
        "chat_template_kwargs must be an object"
    );
    anyhow::ensure!(
        input["chat_template_kwargs"]["use_tts_template"] != false,
        "audio output requires the TTS template"
    );
    input["chat_template_kwargs"]["use_tts_template"] = true.into();
    Ok(std::borrow::Cow::Owned(input))
}
impl Options {
    pub(super) fn parse(input: &Json) -> Result<Self> {
        anyhow::ensure!(
            input
                .get("audio")
                .is_none_or(|v| v.is_null() || v.is_object()),
            "audio must be an object"
        );
        let audio = &input["audio"];
        let format = audio["format"].as_str().unwrap_or("wav");
        anyhow::ensure!(
            ["wav", "pcm16"].contains(&format),
            "audio format must be wav or pcm16"
        );
        anyhow::ensure!(
            audio["voice"].is_null() || audio["voice"] == "default",
            "MiniCPM-o uses voice default or an explicit reference_audio"
        );
        let maximum = speech_sampling::integer(audio, "max_tokens", 2048)?;
        anyhow::ensure!(
            (50..=2048).contains(&maximum),
            "audio max_tokens must be 50..=2048"
        );
        let seconds = audio["reference_seconds"].as_f64().unwrap_or(6.);
        anyhow::ensure!(
            seconds.is_finite() && (1.0..=30.0).contains(&seconds),
            "invalid audio reference_seconds"
        );
        let reference = if audio["reference_audio"].is_null() {
            None
        } else {
            anyhow::ensure!(
                audio["reference_audio"]["format"] == "wav",
                "reference_audio requires WAV"
            );
            let data = audio["reference_audio"]["data"]
                .as_str()
                .context("missing reference_audio.data")?;
            anyhow::ensure!(
                data.len() <= 32 * 1024 * 1024,
                "reference_audio exceeds size limit"
            );
            Some(
                STANDARD
                    .decode(data)
                    .context("decoding reference_audio base64")?,
            )
        };
        let parameters = speech_sampling::parse(audio, maximum)?;
        Ok(Self {
            text: input["modalities"]
                .as_array()
                .is_none_or(|v| v.iter().any(|v| v == "text")),
            format: format.into(),
            maximum,
            seed: audio["seed"]
                .as_u64()
                .or_else(|| input["seed"].as_u64())
                .unwrap_or(42),
            seconds,
            reference,
            parameters,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn modalities_preserve_text_clients_and_validate_template() -> Result<()> {
        assert!(!requested(&json!({}))?);
        assert!(!requested(&json!({"modalities":["text"]}))?);
        assert!(requested(&json!({"modalities":["audio"]}))?);
        assert!(requested(&json!({"modalities":["other"]})).is_err());
        assert!(prepare(&json!({"modalities":["audio"]}), false).is_err());
        let request =
            json!({"modalities":["text","audio"],"chat_template_kwargs":{"enable_thinking":true}});
        let prepared = prepare(&request, true)?;
        assert_eq!(prepared["chat_template_kwargs"]["enable_thinking"], true);
        assert_eq!(prepared["chat_template_kwargs"]["use_tts_template"], true);
        assert!(request["chat_template_kwargs"]["use_tts_template"].is_null());
        assert!(
            prepare(
                &json!({"modalities":["audio"],"chat_template_kwargs":{"use_tts_template":false}}),
                true
            )
            .is_err()
        );
        Ok(())
    }
    #[test]
    fn speech_options_reject_unsupported_output_and_reference() -> Result<()> {
        let options = Options::parse(
            &json!({"modalities":["audio"],"audio":{"format":"pcm16","max_tokens":256}}),
        )?;
        assert!(!options.text);
        assert_eq!(options.maximum, 256);
        assert_eq!(
            options.parameters.policy,
            crate::sampling::speech::Policy::Vllm
        );
        for audio in [
            json!({"format":"mp3"}),
            json!({"voice":"unknown"}),
            json!({"max_tokens":1}),
            json!({"reference_audio":{"format":"wav","data":"invalid!"}}),
        ] {
            assert!(Options::parse(&json!({"audio":audio})).is_err());
        }
        Ok(())
    }
}
