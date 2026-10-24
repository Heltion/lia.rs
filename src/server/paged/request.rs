use super::*;
use anyhow::ensure;

impl Runtime {
    pub(super) fn admit(&mut self) {
        for slot in 0..self.active.len() {
            if self.active.iter().flatten().count() >= self.engine.prefill_batch() {
                break;
            }
            if self.active[slot].is_some() {
                continue;
            }
            if !self.paused.is_empty() {
                if self.resume(slot) {
                    continue;
                }
                break;
            }
            let Some(job) = self.queue.pop_front() else {
                break;
            };
            if job.stream.as_ref().is_some_and(stream::Output::is_closed) {
                continue;
            }
            let prepared = (|| -> Result<_> {
                ensure!(
                    job.input
                        .get("tools")
                        .and_then(Json::as_array)
                        .is_none_or(Vec::is_empty),
                    "MiniCPM5 tool-call response parsing is not yet supported"
                );
                let prompt =
                    crate::chat::render_template_with_bos(&self.template, &job.input, &self.bos)?;
                let tokens = self.tokenizer.encode(&prompt, true)?;
                ensure!(
                    !tokens.is_empty() && tokens.len() < self.context,
                    "prompt exceeds request context"
                );
                let maximum = job
                    .input
                    .get("max_completion_tokens")
                    .or_else(|| job.input.get("max_tokens"))
                    .map(|v| v.as_u64().context("invalid completion token limit"))
                    .transpose()?
                    .unwrap_or(8192);
                let maximum = usize::try_from(maximum)?.min(self.context - tokens.len());
                ensure!(maximum > 0, "completion token limit must be positive");
                let mut input = job.input.clone();
                if input.get("top_k").is_none_or(Json::is_null) {
                    input["top_k"] = json!(0);
                }
                let sampling = sampling::from_request(&input)?;
                let presence = sampling::presence(&input)?;
                let Some((sequence, cached)) = self.attach_for_next(&tokens)? else {
                    return Ok(None);
                };
                Ok(Some((
                    tokens, maximum, sampling, presence, sequence, cached,
                )))
            })();
            match prepared {
                Ok(Some((tokens, maximum, sampling, presence, sequence, cached))) => {
                    if let Some(output) = &job.stream {
                        output.delta(json!({"role":"assistant","content":""}));
                    }
                    let streaming = stream::Text::for_request(&job.input);
                    self.active[slot] = Some(Request {
                        prompt_tokens: tokens.len(),
                        tokens,
                        job,
                        maximum,
                        sampling,
                        presence,
                        sequence,
                        cached,
                        computed: cached,
                        generated: Vec::new(),
                        streaming,
                        first: None,
                        stopped: false,
                        done: false,
                    });
                }
                Ok(None) => {
                    self.queue.push_front(job);
                    break;
                }
                Err(error) => finish_job(job, Err(error)),
            }
        }
    }
}

impl Request {
    pub(super) fn sample(
        &mut self,
        logits: &[f32],
        tokenizer: &Tokenizer,
        eos: &[u32],
    ) -> Result<()> {
        ensure!(
            logits.iter().all(|v| v.is_finite()),
            "nonfinite output logits"
        );
        let mut adjusted = logits.to_vec();
        if self.presence != 0.0 {
            let seen: std::collections::BTreeSet<_> = self.generated.iter().copied().collect();
            for token in seen {
                adjusted[token as usize] -= self.presence;
            }
        }
        let next = if let Some(sampling) = &mut self.sampling {
            sampling.set_position(self.computed.saturating_sub(1) as u64);
            let distribution = sampling.distribution_logits(&adjusted)?;
            sampling.draw(&distribution)
        } else {
            adjusted
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .context("empty output logits")?
                .0 as u32
        };
        self.accept(next, tokenizer, eos)
    }

    pub(super) fn accept(&mut self, next: u32, tokenizer: &Tokenizer, eos: &[u32]) -> Result<()> {
        self.first.get_or_insert_with(Instant::now);
        self.generated.push(next);
        if self.job.input["return_token_ids"] == true {
            if let Some(output) = &self.job.stream {
                output.token(next);
            }
        }
        self.stopped = eos.contains(&next);
        self.done = self.stopped || self.generated.len() >= self.maximum;
        if !self.stopped {
            if let Some(output) = &self.job.stream {
                self.streaming.push(&tokenizer.decode(&[next])?, output)?;
            }
        }
        if !self.done {
            self.tokens.push(next);
        }
        Ok(())
    }

    pub(super) fn response(&mut self, tokenizer: &Tokenizer) -> Result<Json> {
        let now = Instant::now();
        let first = self.first.context("request has no first token")?;
        if let Some(output) = &self.job.stream {
            self.streaming.flush(output)?;
        }
        let raw = String::from_utf8(
            tokenizer
                .decode(&self.generated[..self.generated.len() - usize::from(self.stopped)])?,
        )?;
        let message = crate::chat::parse_with_thinking(
            &raw,
            &json!([]),
            crate::chat::thinking_enabled(&self.job.input),
        )?;
        let prompt_ms = first.duration_since(self.job.submitted).as_secs_f64() * 1000.;
        let predicted_ms = now.duration_since(first).as_secs_f64() * 1000.;
        Ok(
            json!({"id":stream::completion_id(),"created":stream::unix_seconds(),"object":"chat.completion",
            "model":self.job.input["model"].as_str().unwrap_or("lia.rs"),"choices":[{"index":0,"message":message,"finish_reason":if self.stopped {"stop"} else {"length"}}],
            "usage":{"prompt_tokens":self.prompt_tokens,"completion_tokens":self.generated.len(),"total_tokens":self.prompt_tokens+self.generated.len(),"prompt_tokens_details":{"cached_tokens":self.cached}},
            "timings":{"prompt_n":self.prompt_tokens-self.cached,"prompt_ms":prompt_ms,"predicted_n":self.generated.len().saturating_sub(1),"predicted_ms":predicted_ms,"cache_n":self.cached,"mtp_mode":"disabled"}}),
        )
    }
}
