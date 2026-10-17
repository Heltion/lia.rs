use super::*;

impl Session {
    pub(super) fn chat(
        &mut self,
        model: &Model,
        tokenizer: &Tokenizer,
        request: &Json,
    ) -> Result<Json> {
        if request.get("stream").and_then(Json::as_bool) == Some(true) {
            return Err(anyhow!("streaming is not implemented"));
        }
        if request
            .get("temperature")
            .and_then(Json::as_f64)
            .unwrap_or(0.0)
            != 0.0
        {
            return Err(anyhow!("only greedy temperature=0 sampling is implemented"));
        }
        let text = crate::chat::render(model, request)?;
        let prompt = tokenizer.encode(&text, true)?;
        if prompt.is_empty() {
            return Err(anyhow!("empty rendered prompt"));
        }
        let maximum = request
            .get("max_tokens")
            .and_then(Json::as_u64)
            .unwrap_or(8192)
            .min(8192) as usize;
        if maximum == 0 {
            return Err(anyhow!("max_tokens must be positive"));
        }
        let eos = match model.metadata.get("tokenizer.ggml.eos_token_id") {
            Some(Value::Unsigned(x)) => *x as u32,
            _ => return Err(anyhow!("missing EOS token")),
        };
        let start = Instant::now();
        self.engine.trim_scratch();
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
        let mut processed = cached;
        let mut batch_index = 0;
        while processed < prompt.len() {
            let remaining = prompt.len() - processed;
            let count = remaining.min(512);
            let chunk = &prompt[processed..processed + count];
            self.engine.forward_many(chunk, false)?;
            self.tokens.extend_from_slice(chunk);
            processed += count;
            if batch_index % 16 == 0 {
                eprintln!(
                    "prefill {}/{} elapsed_seconds={:.2}",
                    processed,
                    prompt.len(),
                    start.elapsed().as_secs_f64()
                );
            }
            batch_index += 1;
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
        // Include queued MTP recovery work in the measured decode phase.
        self.engine.synchronize()?;
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
        let result = json!({"id":"lia-local","object":"chat.completion","model":"lia.rs","choices":[{"index":0,"message":message,"finish_reason":finish}],"usage":{"prompt_tokens":prompt.len(),"completion_tokens":generated.len(),"total_tokens":prompt.len()+generated.len(),"prompt_tokens_details":{"cached_tokens":cached}},"timings":{"cache_n":cached,"prompt_n":prompt.len()-cached,"prompt_ms":prompt_ms,"predicted_n":generated.len(),"predicted_ms":predicted_ms,"predicted_per_second":generated.len().saturating_sub(1) as f64*1000./predicted_ms,"draft_n":drafts,"draft_n_accepted":accepted,"mtp_mode":"native MTP chain; batched target verification with captured per-position recurrent states","mtp_max_draft":self.engine.max_draft(),"activation_quantization":self.engine.integer_enabled()}});
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
