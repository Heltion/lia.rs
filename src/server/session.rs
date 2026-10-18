use super::*;

impl Session {
    pub(super) fn chat(
        &mut self,
        tokenizer: &Tokenizer,
        request: &Json,
        output: Option<&super::stream::Output>,
    ) -> Result<Json> {
        let mut streaming = super::stream::Text::default();
        if let Some(output) = output {
            output.delta(json!({"role":"assistant","content":""}));
        }
        let mut sampling = super::sampling::from_request(request)?;
        let presence = super::sampling::presence(request)?;
        self.engine.set_presence_penalty(presence)?;
        let mut sampling_info = if let Some(s) = &sampling {
            let (temperature, top_k, top_p) = s.settings();
            json!({"temperature":temperature,"top_k":top_k,"top_p":top_p,"rng":"mt19937","seed":request.get("seed").and_then(Json::as_u64).unwrap_or(42),"mtp_verification":if self.engine.stochastic_mtp() {"stochastic draft rejection and residual sampling"} else {"target sampling and greedy-draft matching"},"draft_top_k":if self.engine.stochastic_mtp(){10}else{1}})
        } else {
            json!({"temperature":0.0,"mtp_verification":"greedy target and draft matching"})
        };
        sampling_info["presence_penalty"] = json!(presence);
        if !self.engine.mtp_enabled() {
            sampling_info["mtp_verification"] = json!("disabled");
            sampling_info["draft_top_k"] = json!(0);
        }
        let mut draft_sampling = if self.engine.stochastic_mtp() {
            sampling
                .as_ref()
                .map(|s| {
                    let (temperature, _, _) = s.settings();
                    crate::sampling::Sampling::new(
                        temperature,
                        10,
                        1.0,
                        request.get("seed").and_then(Json::as_u64).unwrap_or(42) ^ 0x85ebca6b,
                    )
                })
                .transpose()?
        } else {
            None
        };
        let prepared = self.prepare_prompt(tokenizer, request)?;
        let prompt = &prepared.tokens;
        if prompt.is_empty() {
            return Err(anyhow!("empty rendered prompt"));
        }
        let maximum = request
            .get("max_completion_tokens")
            .or_else(|| request.get("max_tokens"))
            .and_then(Json::as_u64)
            .unwrap_or(8192)
            .min(self.engine.capacity().saturating_sub(prompt.len()) as u64)
            as usize;
        if maximum == 0 {
            return Err(anyhow!("max_tokens must be positive"));
        }
        let eos = *self
            .eos_tokens
            .first()
            .ok_or_else(|| anyhow!("missing EOS token"))?;
        let start = Instant::now();
        self.engine.trim_scratch();
        if prepared.sources != self.image_sources {
            self.engine.reset()?;
            self.tokens.clear();
        }
        let common = prompt
            .iter()
            .zip(&self.tokens)
            .take_while(|(a, b)| a == b)
            .count();
        if common != self.tokens.len() {
            if self.checkpoint.position <= common {
                self.engine.restore(&self.checkpoint)?;
                self.tokens.truncate(self.checkpoint.position);
            } else {
                self.engine.reset()?;
                self.tokens.clear();
            }
        }
        if self.tokens.is_empty() && !prepared.images.is_empty() {
            self.engine.prepare_image_inputs(
                &prepared.images,
                prepared
                    .positions
                    .clone()
                    .ok_or_else(|| anyhow!("missing image rotary positions"))?,
            )?;
        } else {
            self.engine.set_position_plan(prepared.positions)?;
        }
        self.image_sources = prepared.sources;
        let cached = self.tokens.len();
        let mut processed = cached;
        let mut batch_index = 0;
        let mut previous_batch = 0;
        let mut prefill_batches = Vec::new();
        while processed < prompt.len() {
            let remaining = prompt.len() - processed;
            let count = self.engine.prefill_chunk(remaining)?;
            if previous_batch != count && previous_batch != 0 {
                self.engine.trim_scratch();
            }
            previous_batch = count;
            prefill_batches.push(count);
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
        if self.engine.prefill_batch() > 512 {
            self.engine.trim_scratch();
        }
        let mut first = if let Some(sampling) = &mut sampling {
            self.engine.current_sample(sampling)?
        } else {
            self.engine.current_token()?
        };
        self.engine.refresh_snapshot(&mut self.checkpoint)?;
        let prompt_ms = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        let mut generated = Vec::new();
        let mut drafts = 0;
        let mut accepted = 0;
        let mut stopped = false;
        while generated.len() < maximum {
            anyhow::ensure!(
                !output.is_some_and(|output| output.is_closed()),
                "stream client disconnected"
            );
            if self.eos_tokens.contains(&first) {
                generated.push(first);
                stopped = true;
                break;
            }
            let (confirmed, next, proposed, matched) = if let (Some(sampling), Some(draft)) =
                (&mut sampling, &mut draft_sampling)
            {
                self.engine.speculate_stochastic(
                    first,
                    maximum - generated.len(),
                    eos,
                    sampling,
                    draft,
                )?
            } else if let Some(sampling) = &mut sampling {
                self.engine
                    .speculate_sampled(first, maximum - generated.len(), eos, sampling)?
            } else {
                self.engine
                    .speculate(first, maximum - generated.len(), eos)?
            };
            drafts += proposed;
            accepted += matched;
            self.tokens.extend_from_slice(&confirmed);
            generated.extend_from_slice(&confirmed);
            if let Some(output) = output {
                streaming.push(
                    &tokenizer.decode(
                        &confirmed[..confirmed.len() - usize::from(confirmed.last() == Some(&eos))],
                    )?,
                    output,
                )?;
            }
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
        if let Some(output) = output {
            streaming.flush(output)?;
        }
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
        let result = json!({"id":super::stream::completion_id(),"created":super::stream::unix_seconds(),"object":"chat.completion","model":request.get("model").and_then(Json::as_str).unwrap_or("lia.rs"),"choices":[{"index":0,"message":message,"finish_reason":finish}],"usage":{"prompt_tokens":prompt.len(),"completion_tokens":generated.len(),"total_tokens":prompt.len()+generated.len(),"prompt_tokens_details":{"cached_tokens":cached}},"timings":{"sampling":sampling_info,"cache_n":cached,"prefill_batches":prefill_batches,"prompt_n":prompt.len()-cached,"prompt_ms":prompt_ms,"predicted_n":generated.len(),"predicted_ms":predicted_ms,"predicted_per_second":generated.len().saturating_sub(1) as f64*1000./predicted_ms,"draft_n":drafts,"draft_n_accepted":accepted,"mtp_mode":if self.engine.mtp_enabled(){"native MTP chain; batched target verification with captured per-position recurrent states"}else{"disabled"},"mtp_max_draft":self.engine.max_draft(),"activation_quantization":self.engine.integer_enabled()}});
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
