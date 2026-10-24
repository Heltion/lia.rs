use super::*;
use crate::inference::PagedRow;

impl Runtime {
    pub(super) fn step(&mut self) -> Result<()> {
        self.pause_for_capacity()?;
        let count = self.active.iter().filter(|r| r.is_some()).count();
        if count == 0 {
            return Ok(());
        }
        let allowance = (self.engine.prefill_batch() / count).max(1);
        let mut rows = Vec::new();
        let mut outputs = Vec::new();
        let mut updates = Vec::new();
        for (slot, request) in self.active.iter_mut().enumerate() {
            let Some(request) = request else {
                continue;
            };
            if request
                .job
                .stream
                .as_ref()
                .is_some_and(stream::Output::is_closed)
            {
                request.done = true;
                continue;
            }
            let end = (request.computed + allowance).min(request.tokens.len());
            let mut computed = request.computed;
            for position in request.computed..end {
                if !self
                    .cache
                    .ensure_position(&mut self.engine, &mut request.sequence, position)?
                {
                    break;
                }
                rows.push(PagedRow {
                    token: request.tokens[position],
                    position: position as u32,
                    sequence: slot as u32,
                });
                computed += 1;
            }
            if computed > request.computed {
                let output = computed == request.tokens.len();
                if output {
                    outputs.push((rows.len() - 1) as u32);
                }
                updates.push((slot, computed, output));
            }
        }
        if !rows.is_empty() {
            let tables: Vec<_> = self
                .active
                .iter()
                .map(|r| {
                    r.as_ref()
                        .map_or_else(Vec::new, |r| r.sequence.pages.clone())
                })
                .collect();
            let output_slots: Vec<_> = updates
                .iter()
                .filter(|(_, _, output)| *output)
                .map(|(slot, _, _)| *slot)
                .collect();
            let fast = self.gpu_sampling
                && !outputs.is_empty()
                && output_slots.iter().all(|&slot| {
                    self.active[slot]
                        .as_ref()
                        .is_some_and(|r| r.presence == 0.0)
                });
            let (logits, next) = if fast {
                let mut samplers: Vec<_> = output_slots
                    .iter()
                    .map(|&slot| {
                        self.active[slot]
                            .as_mut()
                            .expect("active sampling slot")
                            .sampling
                            .take()
                    })
                    .collect();
                let sampled =
                    self.engine
                        .forward_paged_sampled(&rows, &tables, &outputs, &mut samplers);
                for (&slot, sampler) in output_slots.iter().zip(samplers) {
                    self.active[slot]
                        .as_mut()
                        .expect("active sampling slot")
                        .sampling = sampler;
                }
                (Vec::new(), sampled?)
            } else {
                (
                    self.engine.forward_paged(&rows, &tables, &outputs)?,
                    Vec::new(),
                )
            };
            let mut values = logits.chunks_exact(self.engine.vocabulary_size());
            let mut next = next.into_iter();
            for (slot, computed, output) in updates {
                let request = self.active[slot].as_mut().context("lost active request")?;
                request.computed = computed;
                self.cache
                    .seal(&mut request.sequence, &request.tokens, computed)?;
                if output {
                    if fast {
                        request.accept(
                            next.next().context("missing packed sample")?,
                            &self.tokenizer,
                            &self.eos,
                        )?;
                    } else {
                        request.sample(
                            values.next().context("missing packed output")?,
                            &self.tokenizer,
                            &self.eos,
                        )?;
                    }
                }
            }
        } else if self.active.iter().flatten().any(|r| !r.done) {
            anyhow::bail!("active KV scheduler made no progress");
        }
        for slot in 0..self.active.len() {
            if self.active[slot].as_ref().is_some_and(|r| r.done) {
                let mut request = self.active[slot].take().context("lost completed request")?;
                let response = if request.first.is_some() {
                    request.response(&self.tokenizer)
                } else {
                    Err(anyhow!("stream client disconnected"))
                };
                self.cache.release(request.sequence)?;
                finish_job(request.job, response);
            }
        }
        Ok(())
    }
}
