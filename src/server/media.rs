//! Validate multimodal sources while preserving their order for prefix identity.
use super::*;
use anyhow::{Context, ensure};

pub(super) struct Sources {
    pub images: Vec<String>,
    pub audios: Vec<String>,
    pub ordered: Vec<String>,
}

pub(super) fn sources(request: &Json) -> Result<Sources> {
    let mut sources = Sources {
        images: Vec::new(),
        audios: Vec::new(),
        ordered: Vec::new(),
    };
    for message in request["messages"]
        .as_array()
        .context("expected messages array")?
    {
        let Some(parts) = message["content"].as_array() else {
            continue;
        };
        for part in parts {
            let (source, image) = match part["type"].as_str() {
                Some("text") => {
                    ensure!(part["text"].is_string(), "expected content text");
                    continue;
                }
                Some("image_url") => (
                    part["image_url"]["url"]
                        .as_str()
                        .context("expected image_url.url")?
                        .to_owned(),
                    true,
                ),
                Some("audio_url") => (
                    part["audio_url"]["url"]
                        .as_str()
                        .context("expected audio_url.url")?
                        .to_owned(),
                    false,
                ),
                Some("input_audio") => {
                    ensure!(
                        part["input_audio"]["format"] == "wav",
                        "input_audio currently requires WAV"
                    );
                    let data = part["input_audio"]["data"]
                        .as_str()
                        .context("expected input_audio.data")?;
                    (format!("data:audio/wav;base64,{data}"), false)
                }
                _ => anyhow::bail!("unsupported MiniCPM-o content part"),
            };
            ensure!(
                source.starts_with("data:"),
                "media requires a base64 data URL"
            );
            if image {
                sources.images.push(source.clone());
            } else {
                sources.audios.push(source.clone());
            }
            sources.ordered.push(source);
        }
    }
    Ok(sources)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mixed_media_identity_preserves_order_and_validates_audio() -> Result<()> {
        let request = json!({"messages":[{"content":[{"type":"input_audio","input_audio":{"format":"wav","data":"sample"}},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,image"}}]}]});
        let parsed = sources(&request)?;
        assert_eq!(parsed.audios, vec!["data:audio/wav;base64,sample"]);
        assert_eq!(parsed.images, vec!["data:image/png;base64,image"]);
        assert_eq!(parsed.ordered[0], parsed.audios[0]);
        assert_eq!(parsed.ordered[1], parsed.images[0]);
        assert!(sources(&json!({"messages":[{"content":[{"type":"input_audio","input_audio":{"format":"mp3","data":"x"}}]}]})).is_err());
        Ok(())
    }
}
