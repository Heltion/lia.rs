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
    multimodal: bool,
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
            multimodal: request["modalities"]
                .as_array()
                .is_some_and(|v| v.iter().any(|v| v == "audio")),
        };
        let events = futures_util::stream::unfold(receiver, |mut receiver| async {
            receiver.recv().await.map(|data| {
                (
                    Ok::<_, Infallible>(axum::response::sse::Event::default().data(data)),
                    receiver,
                )
            })
        });
        // Prefill and buffered tool arguments can outlast HTTP body-idle limits.
        // SSE comments keep the transport live without adding model output.
        let response = axum::response::Sse::new(events)
            .keep_alive(axum::response::sse::KeepAlive::new())
            .into_response();
        (output, response)
    }

    fn send(&self, value: Json) {
        let _ = self.sender.send(value.to_string());
    }

    fn chunk(&self, choices: Json) -> Json {
        let mut chunk = json!({"id":self.id,"object":"chat.completion.chunk","created":self.created,"model":self.model,"choices":choices});
        if self.multimodal {
            chunk["modality"] = "text".into();
        }
        chunk
    }

    pub(super) fn is_closed(&self) -> bool {
        self.sender.is_closed()
    }

    pub(super) fn delta(&self, delta: Json) {
        self.send(self.chunk(json!([{"index":0,"delta":delta,"finish_reason":null}])));
    }

    pub(super) fn token(&self, token: u32) {
        self.send(
            self.chunk(json!([{"index":0,"delta":{},"finish_reason":null,"token_ids":[token]}])),
        );
    }

    pub(super) fn audio_id(&self) -> String {
        format!("audio-{}", self.id)
    }

    pub(super) fn audio(
        &self,
        bytes: &[u8],
        format: &str,
        sample_offset: usize,
        code_n: usize,
        last: bool,
    ) -> Result<()> {
        use base64::{Engine as _, engine::general_purpose::STANDARD};
        let audio = json!({"id":self.audio_id(),"data":STANDARD.encode(bytes),"format":format,"sample_rate":24000,"sample_offset":sample_offset,"confirmed_code_n":code_n,"segment_final":last});
        let mut chunk = self.chunk(json!([{"index":0,"delta":{"audio":audio,"content":audio["data"]},"finish_reason":null}]));
        chunk["modality"] = "audio".into();
        self.sender
            .send(chunk.to_string())
            .context("sending incremental audio")?;
        Ok(())
    }

    pub(super) fn finish(&self, response: &Result<Json>) {
        match response {
            Ok(response) => {
                if let Some(audio) = response["choices"][0]["message"].get("audio") {
                    let mut audio = audio.clone();
                    if response["timings"]["speech"]["audio_delivery"] == "incremental" {
                        if let Some(object) = audio.as_object_mut() {
                            object.remove("data");
                        }
                    }
                    let mut delta = json!({"audio":audio});
                    if let Some(data) = audio.get("data") {
                        delta["content"] = data.clone();
                    }
                    let mut chunk =
                        self.chunk(json!([{"index":0,"delta":delta,"finish_reason":null}]));
                    chunk["modality"] = "audio".into();
                    self.send(chunk);
                }
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
    started: bool,
}

impl Text {
    pub(super) fn for_request(request: &Json) -> Self {
        Self::with_thinking(crate::chat::thinking_enabled(request))
    }
    pub(super) fn with_thinking(thinking: bool) -> Self {
        Self {
            content: !thinking,
            ..Self::default()
        }
    }
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
            // Match the non-streamed response parser without trimming whitespace
            // inside a channel or splitting a Unicode character.
            if !self.started {
                let leading = text.len() - text.trim_start().len();
                if leading > 0 {
                    self.pending.drain(..leading);
                    continue;
                }
                if !self.content {
                    if text.starts_with("<think>") {
                        self.pending.drain(.."<think>".len());
                        continue;
                    }
                    if !last && "<think>".starts_with(text) {
                        return Ok(());
                    }
                }
                self.started = !text.is_empty();
            }
            let implicit = !self.content
                && text
                    .find("<tool_call>")
                    .is_some_and(|call| text.find("</think>").is_none_or(|end| call < end));
            let marker = if self.content || implicit {
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
                        text[..position].trim_end(),
                    );
                }
                self.pending.drain(..position + marker.len());
                if self.content || implicit {
                    self.tools = true;
                    self.pending.clear();
                    return Ok(());
                }
                self.content = true;
                self.started = false;
                continue;
            }
            // Hold a possible split delimiter and incomplete UTF-8 until the next token.
            let held = if last {
                0
            } else {
                (1.."<tool_call>".len())
                    .rev()
                    .find(|&n| {
                        let markers = if self.content {
                            &["<tool_call>"][..]
                        } else {
                            &["</think>", "<tool_call>"][..]
                        };
                        markers
                            .iter()
                            .any(|marker| n < marker.len() && text.ends_with(&marker[..n]))
                    })
                    .unwrap_or(0)
            };
            let length = text[..valid - held].trim_end().len();
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
            if last {
                self.pending.clear();
            }
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incremental_audio_finish_sends_transcript_without_replaying_samples() -> Result<()> {
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let output = Output {
            sender,
            id: "test".into(),
            model: "test".into(),
            created: 0,
            usage: true,
            multimodal: true,
        };
        output.audio(&[1, 2], "pcm16", 0, 28, false)?;
        output.finish(&Ok(json!({"choices":[{"message":{"audio":{"id":output.audio_id(),"data":"REPLAY","transcript":"hello"}},"finish_reason":"stop"}],"timings":{"speech":{"audio_delivery":"incremental"}}})));
        let first: Json = serde_json::from_str(&receiver.try_recv()?)?;
        assert_eq!(first["choices"][0]["delta"]["audio"]["data"], "AQI=");
        let final_audio: Json = serde_json::from_str(&receiver.try_recv()?)?;
        let delta = &final_audio["choices"][0]["delta"];
        assert_eq!(delta["audio"]["transcript"], "hello");
        assert!(delta["audio"].get("data").is_none() && delta.get("content").is_none());
        Ok(())
    }

    #[test]
    fn idle_stream_preserves_transport_without_semantic_events() -> Result<()> {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?
            .block_on(async {
                let (output, response) = Output::new(&json!({"model":"test"}));
                let finish = tokio::spawn(async move {
                    tokio::time::sleep(std::time::Duration::from_secs(16)).await;
                    output.finish(&Ok(json!({"choices":[{"finish_reason":"stop"}]})));
                });
                let body = axum::body::to_bytes(response.into_body(), 65536).await?;
                finish.await?;
                let wire = std::str::from_utf8(&body)?;
                assert!(wire.contains(":\n\n"), "expected a keepalive comment");
                assert_eq!(wire.lines().filter(|s| s.starts_with("data:")).count(), 2);
                assert!(wire.contains("[DONE]"));
                Ok(())
            })
    }

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

    #[test]
    fn streamed_channel_whitespace_matches_nonstreamed_content() -> Result<()> {
        for (raw, width) in [
            " \n思考 中\t \n</think>\n\n  B words\n keep indent\t \n",
            " \n思考 尚未结束\t \n",
            " \n<think>\n<think>思考 尚未结束\t \n",
            "<think>思考</think>\nanswer",
            "```json\n{\"value\":\"42\"}\n```",
            "先查询<tool_call><function=f></function></tool_call>",
        ]
        .into_iter()
        .flat_map(|raw| [1, 2, 3, 7, 128].into_iter().map(move |width| (raw, width)))
        {
            let mut stream = Text::default();
            let mut reasoning = String::new();
            let mut content = String::new();
            let mut emit = |key: &str, value: &str| {
                if key == "content" {
                    content.push_str(value);
                } else {
                    reasoning.push_str(value);
                }
            };
            for bytes in raw.as_bytes().chunks(width) {
                stream.consume(bytes, false, &mut emit)?;
            }
            stream.consume(&[], true, &mut emit)?;
            let parsed = crate::chat::parse(raw, &json!([]))?;
            assert_eq!(content, parsed["content"].as_str().unwrap());
            assert_eq!(reasoning, parsed["reasoning_content"].as_str().unwrap());
        }
        Ok(())
    }
}
