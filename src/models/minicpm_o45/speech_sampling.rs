//! MiniCPM-o codec settings and explicit audio-option validation.
use crate::sampling::speech::{Parameters, Policy};
use anyhow::{Context, Result, ensure};
use serde_json::Value;

fn number(audio: &Value, name: &str, default: f64) -> Result<f64> {
    match audio.get(name).filter(|v| !v.is_null()) {
        Some(value) => value
            .as_f64()
            .with_context(|| format!("audio.{name} must be a number")),
        None => Ok(default),
    }
}

pub(crate) fn integer(audio: &Value, name: &str, default: usize) -> Result<usize> {
    match audio.get(name).filter(|v| !v.is_null()) {
        Some(value) => usize::try_from(
            value
                .as_u64()
                .with_context(|| format!("audio.{name} must be a nonnegative integer"))?,
        )
        .with_context(|| format!("audio.{name} exceeds the supported range")),
        None => Ok(default),
    }
}

pub(crate) fn parse(audio: &Value, maximum: usize) -> Result<Parameters> {
    let mut parameters = Parameters::default();
    let policy = match audio.get("sampling_policy").filter(|v| !v.is_null()) {
        Some(value) => value
            .as_str()
            .context("audio.sampling_policy must be a string")?,
        None => "vllm",
    };
    parameters.policy = match policy {
        "vllm" => Policy::Vllm,
        "checkpoint" => Policy::Checkpoint,
        _ => anyhow::bail!("unknown audio sampling_policy"),
    };
    if parameters.policy == Policy::Vllm {
        // Match the pinned deployment and the native realtime codec settings.
        // The primitive checkpoint defaults remain available under that policy.
        parameters.top_k = 100;
        parameters.top_p = 0.8;
    }
    parameters.temperature =
        number(audio, "temperature", f64::from(parameters.temperature))? as f32;
    parameters.top_p = number(audio, "top_p", parameters.top_p)?;
    parameters.top_k = integer(audio, "top_k", parameters.top_k)?;
    parameters.repetition_penalty = number(
        audio,
        "repetition_penalty",
        f64::from(parameters.repetition_penalty),
    )? as f32;
    parameters.min_tokens = integer(audio, "min_tokens", parameters.min_tokens)?;
    ensure!(
        parameters.min_tokens <= maximum,
        "audio.min_tokens exceeds audio.max_tokens"
    );
    parameters
        .validate()
        .context("validating audio sampling parameters")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn deployment_defaults_and_explicit_overrides_preserve_policy() -> Result<()> {
        let native = parse(&json!({}), 2048)?;
        assert_eq!(native.policy, Policy::Vllm);
        assert_eq!(
            (native.top_k, native.top_p, native.min_tokens),
            (100, 0.8, 50)
        );
        let checkpoint = parse(&json!({"sampling_policy":"checkpoint"}), 2048)?;
        assert_eq!(checkpoint.policy, Policy::Checkpoint);
        assert_eq!((checkpoint.top_k, checkpoint.top_p), (25, 0.85));
        let explicit = parse(
            &json!({"temperature":0.6,"top_k":25,"top_p":0.85,
            "repetition_penalty":1.1,"min_tokens":12}),
            256,
        )?;
        assert_eq!(explicit.policy, Policy::Vllm);
        assert_eq!(
            (explicit.temperature, explicit.top_k, explicit.top_p),
            (0.6, 25, 0.85)
        );
        assert_eq!(
            (explicit.repetition_penalty, explicit.min_tokens),
            (1.1, 12)
        );
        assert_eq!(parse(&json!({"top_k":0,"min_tokens":0}), 50)?.top_k, 0);
        Ok(())
    }

    #[test]
    fn invalid_codec_settings_fail_before_synthesis() {
        for value in [
            json!({"temperature":0}),
            json!({"temperature":1e100}),
            json!({"top_p":0}),
            json!({"top_p":1.01}),
            json!({"top_p":"0.8"}),
            json!({"top_k":-1}),
            json!({"top_k":1.5}),
            json!({"top_k":6563}),
            json!({"repetition_penalty":0}),
            json!({"min_tokens":257}),
            json!({"sampling_policy":true}),
            json!({"sampling_policy":"other"}),
        ] {
            assert!(parse(&value, 256).is_err(), "accepted {value}");
        }
        assert!(integer(&json!({"max_tokens":-1}), "max_tokens", 2048).is_err());
    }
}
