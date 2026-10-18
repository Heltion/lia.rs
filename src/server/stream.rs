//! OpenAI event framing and incremental separation of model reasoning and text.
use super::*;
use axum::response::IntoResponse;
use std::{
    convert::Infallible,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub(super) fn unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub(super) fn completion_id() -> String {
    let time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!(
        "chatcmpl-lia-{time}-{}",
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    )
}

pub(super) struct Output {
    sender: tokio::sync::mpsc::UnboundedSender<String>,
    id: String,
    model: String,
    created: u64,
    usage: bool,
}

impl Output {
    pub(super) fn new(request: &Json) -> (Self, axum::response::Response) {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        let output = Self {
            sender,
            id: completion_id(),
            model: request["model"].as_str().unwrap_or("lia.rs").to_owned(),
            created: unix_seconds(),
            usage: request["stream_options"]["include_usage"] == true,
        };
        let events = futures_util::stream::unfold(receiver, |mut receiver| async {
            receiver.recv().await.map(|data| {
                (
                    Ok::<_, Infallible>(axum::response::sse::Event::default().data(data)),
                    receiver,
                )
            })
        });
        (output, axum::response::Sse::new(events).into_response())
    }

    fn send(&self, value: Json) {
        let _ = self.sender.send(value.to_string());
    }

    fn chunk(&self, choices: Json) -> Json {
        json!({"id":self.id,"object":"chat.completion.chunk","created":self.created,"model":self.model,"choices":choices})
    }

    pub(super) fn is_closed(&self) -> bool {
        self.sender.is_closed()
    }

    pub(super) fn delta(&self, delta: Json) {
        self.send(self.chunk(json!([{"index":0,"delta":delta,"finish_reason":null}])));
    }

    pub(super) fn finish(&self, response: &Result<Json>) {
        match response {
            Ok(response) => {
                if let Some(calls) = response["choices"][0]["message"]["tool_calls"].as_array() {
                    for (index, call) in calls.iter().enumerate() {
                        let mut call = call.clone();
                        call["index"] = json!(index);
                        call["id"] = json!(format!("{}_call_{index}", self.id));
                        self.delta(json!({"tool_calls":[call]}));
                    }
                }
                self.send(self.chunk(json!([{"index":0,"delta":{},"finish_reason":response["choices"][0]["finish_reason"]}])));
                if self.usage {
                    let mut chunk = self.chunk(json!([]));
                    chunk["usage"] = response["usage"].clone();
                    chunk["timings"] = response["timings"].clone();
                    self.send(chunk);
                }
            }
            Err(error) => self
                .send(json!({"error":{"message":format!("{error:#}"),"type":"inference_error"}})),
        }
        let _ = self.sender.send("[DONE]".into());
    }
}

#[derive(Default)]
pub(super) struct Text {
    pending: Vec<u8>,
    content: bool,
    tools: bool,
}

impl Text {
    pub(super) fn push(&mut self, bytes: &[u8], output: &Output) -> Result<()> {
        self.consume(bytes, false, |key, text| output.delta(json!({key:text})))
    }

    pub(super) fn flush(&mut self, output: &Output) -> Result<()> {
        self.consume(&[], true, |key, text| output.delta(json!({key:text})))
    }

    fn consume(
        &mut self,
        bytes: &[u8],
        last: bool,
        mut emit: impl FnMut(&str, &str),
    ) -> Result<()> {
        if self.tools {
            return Ok(());
        }
        self.pending.extend_from_slice(bytes);
        loop {
            let valid = match std::str::from_utf8(&self.pending) {
                Ok(text) => text.len(),
                Err(error) if error.error_len().is_none() && !last => error.valid_up_to(),
                Err(error) => return Err(error.into()),
            };
            let text = std::str::from_utf8(&self.pending[..valid])?;
            let marker = if self.content {
                "<tool_call>"
            } else {
                "</think>"
            };
            if let Some(position) = text.find(marker) {
                if position > 0 {
                    emit(
                        if self.content {
                            "content"
                        } else {
                            "reasoning_content"
                        },
                        &text[..position],
                    );
                }
                self.pending.drain(..position + marker.len());
                if self.content {
                    self.tools = true;
                    self.pending.clear();
                    return Ok(());
                }
                self.content = true;
                continue;
            }
            // Hold a possible split delimiter and incomplete UTF-8 until the next token.
            let held = if last {
                0
            } else {
                (1..marker.len())
                    .rev()
                    .find(|&n| text.ends_with(&marker[..n]))
                    .unwrap_or(0)
            };
            let length = valid - held;
            if length > 0 {
                emit(
                    if self.content {
                        "content"
                    } else {
                        "reasoning_content"
                    },
                    &text[..length],
                );
                self.pending.drain(..length);
            }
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delimiters_and_unicode_across_every_byte() -> Result<()> {
        let mut text = Text::default();
        let mut emitted = Vec::new();
        for byte in "思考</think>answer<tool_call><function=bash>".bytes() {
            text.consume(&[byte], false, |key, value| {
                emitted.push((key.to_owned(), value.to_owned()))
            })?;
        }
        let reasoning: String = emitted
            .iter()
            .filter(|(key, _)| key == "reasoning_content")
            .map(|(_, value)| value.as_str())
            .collect();
        let content: String = emitted
            .iter()
            .filter(|(key, _)| key == "content")
            .map(|(_, value)| value.as_str())
            .collect();
        assert_eq!(reasoning, "思考");
        assert_eq!(content, "answer");
        assert!(text.tools);
        Ok(())
    }
}
