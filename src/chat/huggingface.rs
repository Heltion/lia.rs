//! Hugging Face stores templates either independently or in tokenizer metadata.
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::{fs, path::Path};

/// Adapt OpenAI content parts for checkpoints whose template expects strings.
pub fn string_messages(request: &Value, image_marker: &str) -> Result<Value> {
    let mut request = request.clone();
    let messages = request["messages"]
        .as_array_mut()
        .context("expected messages array")?;
    for message in messages {
        if let Some(parts) = message["content"].as_array() {
            let mut pieces = Vec::new();
            for part in parts {
                match part["type"].as_str() {
                    Some("text") => pieces.push(
                        part["text"]
                            .as_str()
                            .context("expected content text")?
                            .to_owned(),
                    ),
                    Some("image_url") => pieces.push(image_marker.to_owned()),
                    Some("input_audio" | "audio_url") => pieces.push("<|audio|>".to_owned()),
                    _ => anyhow::bail!("unsupported string-template content part"),
                }
            }
            message["content"] = Value::String(pieces.join("\n"));
        } else if message["role"] == "assistant"
            && message["content"].is_null()
            && message["tool_calls"].is_array()
        {
            message["content"] = Value::String(String::new());
        }
    }
    Ok(request)
}

pub fn huggingface_template(directory: impl AsRef<Path>) -> Result<String> {
    let directory = directory.as_ref();
    let path = directory.join("chat_template.jinja");
    match fs::read_to_string(&path) {
        Ok(template) => {
            ensure!(
                !template.trim().is_empty(),
                "empty template {}",
                path.display()
            );
            return Ok(template);
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).with_context(|| format!("reading {}", path.display())),
    }
    let path = directory.join("tokenizer_config.json");
    let config: Value = serde_json::from_reader(
        fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?,
    )
    .with_context(|| format!("reading {}", path.display()))?;
    embedded_template(&config["chat_template"])
        .with_context(|| format!("selecting default chat template in {}", path.display()))
}

fn embedded_template(value: &Value) -> Result<String> {
    let template = if let Some(text) = value.as_str() {
        text
    } else {
        let templates = value
            .as_array()
            .context("expected template text or named templates")?;
        let defaults: Vec<_> = templates
            .iter()
            .filter(|v| v["name"] == "default")
            .collect();
        ensure!(
            defaults.len() == 1,
            "expected one default named chat template"
        );
        defaults[0]["template"]
            .as_str()
            .context("default template is not text")?
    };
    ensure!(!template.trim().is_empty(), "empty embedded chat template");
    Ok(template.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_content_keeps_order_and_original_request() -> Result<()> {
        let request = serde_json::json!({"messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":"data:image/png;base64,x"}},
            {"type":"text","text":"Describe it."}]}]});
        let adapted = string_messages(&request, "IMAGE")?;
        assert_eq!(adapted["messages"][0]["content"], "IMAGE\nDescribe it.");
        assert!(request["messages"][0]["content"].is_array());
        Ok(())
    }

    #[test]
    fn embedded_template_uses_explicit_default() -> Result<()> {
        assert_eq!(embedded_template(&serde_json::json!("plain"))?, "plain");
        assert_eq!(
            embedded_template(&serde_json::json!([
                {"name":"tool_use","template":"tools"}, {"name":"default","template":"normal"}
            ]))?,
            "normal"
        );
        assert!(embedded_template(&Value::Null).is_err());
        assert!(embedded_template(&serde_json::json!(" ")).is_err());
        assert!(
            embedded_template(&serde_json::json!([
                {"name":"default","template":"one"},{"name":"default","template":"two"}
            ]))
            .is_err()
        );
        Ok(())
    }
}
