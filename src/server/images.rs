use super::*;
use crate::vision::{ImageGrid, PositionPlan, PreparedImage};
use anyhow::ensure;

pub(super) struct Prompt {
    pub tokens: Vec<u32>,
    pub sources: Vec<String>,
    pub images: Vec<(usize, PreparedImage)>,
    pub positions: Option<PositionPlan>,
}

fn image_sources(request: &Json) -> Result<Vec<String>> {
    let mut sources = Vec::new();
    let messages = request["messages"]
        .as_array()
        .ok_or_else(|| anyhow!("expected messages array"))?;
    for message in messages {
        let Some(parts) = message["content"].as_array() else {
            continue;
        };
        for part in parts {
            match part["type"].as_str() {
                Some("text") => {
                    ensure!(part["text"].is_string(), "expected content text");
                }
                Some("image_url") => {
                    let url = part["image_url"]["url"]
                        .as_str()
                        .ok_or_else(|| anyhow!("expected image_url.url"))?;
                    ensure!(
                        url.starts_with("data:"),
                        "image_url must contain PNG/JPEG base64 data"
                    );
                    sources.push(url.to_owned());
                }
                _ => return Err(anyhow!("unsupported chat content part")),
            }
        }
    }
    Ok(sources)
}

impl Session {
    pub(super) fn prepare_prompt(
        &self,
        model: &Model,
        tokenizer: &Tokenizer,
        request: &Json,
    ) -> Result<Prompt> {
        let sources = image_sources(request)?;
        let text = crate::chat::render(model, request)?;
        let tokens = tokenizer.encode(&text, true)?;
        if sources.is_empty() {
            return Ok(Prompt {
                tokens,
                sources,
                images: Vec::new(),
                positions: None,
            });
        }
        let processor = self
            .processor
            .as_ref()
            .ok_or_else(|| anyhow!("image input requires --mmproj"))?;
        let marker = tokenizer.encode("<|image_pad|>", true)?;
        ensure!(marker.len() == 1, "image marker must be one special token");
        ensure!(
            tokens.iter().filter(|t| **t == marker[0]).count() == sources.len(),
            "rendered image markers do not match image inputs"
        );
        let mut expanded = Vec::new();
        let mut images = Vec::new();
        let mut positions = PositionPlan::new();
        for token in tokens {
            if token == marker[0] {
                let image = processor.prepare_data_url(&sources[images.len()])?;
                let grid = ImageGrid::new(image.width, image.height)?;
                let start = positions.append_image(grid)?;
                expanded.extend(std::iter::repeat_n(token, grid.token_count()));
                images.push((start, image));
            } else {
                positions.append_text(1)?;
                expanded.push(token);
            }
        }
        ensure!(
            expanded.len() <= self.engine.capacity(),
            "image prompt exceeds context capacity"
        );
        Ok(Prompt {
            tokens: expanded,
            sources,
            images,
            positions: Some(positions),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_parts_preserve_image_order_and_reject_unsupported_inputs() -> Result<()> {
        let request = json!({"messages":[{"role":"user","content":[
            {"type":"text","text":"before"},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,one"}},
            {"type":"text","text":"between"},
            {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,two"}}]}]});
        assert_eq!(
            image_sources(&request)?,
            vec!["data:image/png;base64,one", "data:image/jpeg;base64,two"]
        );
        assert!(
            image_sources(&json!({"messages":[{"content":[{"type":"input_audio"}]}]})).is_err()
        );
        assert!(image_sources(&json!({"messages":[{"content":[{"type":"text"}]}]})).is_err());
        assert!(image_sources(&json!({"messages":[{"content":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}]})).is_err());
        assert!(image_sources(&json!({"messages":[{"content":"ordinary text"}]}))?.is_empty());
        Ok(())
    }
}
