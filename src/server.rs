//! Serial local chat endpoint; model execution remains in the Rust CUDA engine.
use crate::{
    gguf::{Model, Value},
    inference::{Engine, Snapshot},
    tokenizer::Tokenizer,
};
use serde_json::{Value as Json, json};
use std::{io::Read, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
struct Session {
    engine: Engine,
    tokens: Vec<u32>,
    checkpoint: Option<Snapshot>,
}

impl Session {
    fn chat(&mut self, model: &Model, tokenizer: &Tokenizer, request: &Json) -> Result<Json> {
        if request.get("stream").and_then(Json::as_bool) == Some(true) {
            return Err("streaming is not implemented".into());
        }
        if request
            .get("temperature")
            .and_then(Json::as_f64)
            .unwrap_or(0.0)
            != 0.0
        {
            return Err("only greedy temperature=0 sampling is implemented".into());
        }
        let text = crate::chat::render(model, request)?;
        let prompt = tokenizer.encode(&text, true)?;
        if prompt.is_empty() {
            return Err("empty rendered prompt".into());
        }
        let maximum = request
            .get("max_tokens")
            .and_then(Json::as_u64)
            .unwrap_or(8192)
            .min(8192) as usize;
        if maximum == 0 {
            return Err("max_tokens must be positive".into());
        }
        let eos = match model.metadata.get("tokenizer.ggml.eos_token_id") {
            Some(Value::Unsigned(x)) => *x as u32,
            _ => return Err("missing EOS token".into()),
        };
        let start = Instant::now();
        let common = prompt
            .iter()
            .zip(&self.tokens)
            .take_while(|(a, b)| a == b)
            .count();
        if common != self.tokens.len() {
            if let Some(checkpoint) = self.checkpoint.as_ref().filter(|s| s.position <= common) {
                self.engine.restore(checkpoint)?;
                self.tokens.truncate(checkpoint.position);
            } else {
                self.engine.reset()?;
                self.tokens.clear();
            }
        }
        let cached = self.tokens.len();
        for (i, chunk) in prompt[cached..].chunks(8).enumerate() {
            self.engine.forward_many(chunk, false)?;
            self.tokens.extend_from_slice(chunk);
            if i % 16 == 0 {
                eprintln!(
                    "prefill {}/{} elapsed_seconds={:.2}",
                    self.tokens.len(),
                    prompt.len(),
                    start.elapsed().as_secs_f64()
                );
            }
        }
        let mut first = self.engine.current_token()?;
        self.checkpoint = Some(self.engine.refresh_snapshot(self.checkpoint.take())?);
        let prompt_ms = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        let mut generated = Vec::new();
        let mut drafts = 0;
        let mut accepted = 0;
        let mut stopped = false;
        while generated.len() < maximum {
            if first == eos {
                generated.push(first);
                stopped = true;
                break;
            }
            let (confirmed, next, proposed, matched) =
                self.engine
                    .speculate(first, maximum - generated.len(), eos)?;
            drafts += proposed;
            accepted += matched;
            self.tokens.extend_from_slice(&confirmed);
            generated.extend_from_slice(&confirmed);
            first = next;
            if confirmed.last() == Some(&eos) {
                stopped = true;
                break;
            }
            if generated.len() % 32 < 4 {
                eprintln!(
                    "generation tokens={} elapsed_seconds={:.2} mtp_accepted={accepted}/{drafts}",
                    generated.len(),
                    start.elapsed().as_secs_f64()
                );
            }
        }
        let predicted_ms = start.elapsed().as_secs_f64() * 1000.;
        let raw = String::from_utf8(
            tokenizer.decode(&generated[..generated.len() - usize::from(stopped)])?,
        )?;
        let tools = request.get("tools").cloned().unwrap_or(json!([]));
        let message = crate::chat::parse(&raw, &tools)?;
        let has_tools = message
            .get("tool_calls")
            .and_then(Json::as_array)
            .is_some_and(|a| !a.is_empty());
        let finish = if !stopped {
            "length"
        } else if has_tools {
            "tool_calls"
        } else {
            "stop"
        };
        let result = json!({"id":"lia-local","object":"chat.completion","model":"lia.rs","choices":[{"index":0,"message":message,"finish_reason":finish}],"usage":{"prompt_tokens":prompt.len(),"completion_tokens":generated.len(),"total_tokens":prompt.len()+generated.len(),"prompt_tokens_details":{"cached_tokens":cached}},"timings":{"cache_n":cached,"prompt_n":prompt.len()-cached,"prompt_ms":prompt_ms,"predicted_n":generated.len(),"predicted_ms":predicted_ms,"predicted_per_second":generated.len() as f64*1000./predicted_ms,"draft_n":drafts,"draft_n_accepted":accepted,"mtp_mode":"native MTP chain; batched target verification with captured per-position recurrent states","mtp_max_draft":self.engine.max_draft(),"activation_quantization":self.engine.integer_enabled()}});
        eprintln!(
            "request complete: prompt_tokens={} generation_tokens={} prompt_seconds={:.3} generation_seconds={:.3}",
            prompt.len(),
            generated.len(),
            prompt_ms / 1000.,
            predicted_ms / 1000.
        );
        Ok(result)
    }
}
pub fn run(path: &str, port: u16, capacity: usize) -> Result<()> {
    let model = Model::open(path)?;
    let tokenizer = Tokenizer::from_model(&model)?;
    let engine = Engine::load(path, capacity)?;
    let mut session = Session {
        engine,
        tokens: Vec::new(),
        checkpoint: None,
    };
    let server = tiny_http::Server::http(("127.0.0.1", port)).map_err(|e| e.to_string())?;
    eprintln!("lia.rs listening on 127.0.0.1:{port}; thinking enabled; built-in MTP enabled");
    for mut request in server.incoming_requests() {
        let response = if request.url() == "/health" {
            Ok(
                json!({"status":"ok","backend":"lia.rs","thinking":true,"mtp":true,"mtp_max_draft":session.engine.max_draft(),"target_verify_max_batch":session.engine.max_draft()+1,"mtp_state_restore":"captured per-position recurrent states","prefill_batch":8,"kv_cache":"f16","activation_quantization":session.engine.integer_enabled(),"matrix_split_k":session.engine.matrix_layout().0,"matrix_row_group":session.engine.matrix_layout().1,"attention":"partitioned_online_softmax","attention_tile":128}),
            )
        } else {
            let mut body = String::new();
            std::io::Read::take(request.as_reader(), 16 * 1024 * 1024).read_to_string(&mut body)?;
            let parsed: std::result::Result<Json, _> = serde_json::from_str(&body);
            match parsed {
                Ok(input) => match request.url() {
                    "/v1/chat/completions" => session.chat(&model, &tokenizer, &input),
                    "/apply-template" => {
                        crate::chat::render(&model, &input).map(|prompt| json!({"prompt":prompt}))
                    }
                    "/tokenize" => tokenizer
                        .encode(
                            input.get("content").and_then(Json::as_str).unwrap_or(""),
                            input
                                .get("parse_special")
                                .and_then(Json::as_bool)
                                .unwrap_or(true),
                        )
                        .map(|tokens| json!({"tokens":tokens})),
                    _ => Err("unknown endpoint".into()),
                },
                Err(error) => Err(error.into()),
            }
        };
        let (status, body) = match response {
            Ok(body) => (200, body),
            Err(error) => {
                eprintln!("request failed: {error}");
                (400, json!({"error":{"message":error.to_string()}}))
            }
        };
        let response = tiny_http::Response::from_string(serde_json::to_string(&body)?)
            .with_status_code(status)
            .with_header(
                tiny_http::Header::from_bytes("Content-Type", "application/json")
                    .map_err(|_| "invalid response header")?,
            );
        request.respond(response)?;
    }
    Ok(())
}
