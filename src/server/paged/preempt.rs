//! Requests grow KV on demand; paused sequences retain tokens and sampler state.
use super::*;

impl Runtime {
    pub(super) fn attach_for_next(&mut self, tokens: &[u32]) -> Result<Option<(Sequence, usize)>> {
        let (mut sequence, cached) = self.cache.attach(&mut self.engine, tokens)?;
        match self
            .cache
            .ensure_position(&mut self.engine, &mut sequence, cached)
        {
            Ok(true) => Ok(Some((sequence, cached))),
            Ok(false) => {
                self.cache.release(sequence)?;
                Ok(None)
            }
            Err(error) => {
                self.cache.release(sequence)?;
                Err(error)
            }
        }
    }

    pub(super) fn resume(&mut self, slot: usize) -> bool {
        let Some(request) = self.paused.front() else {
            return false;
        };
        if request
            .job
            .stream
            .as_ref()
            .is_some_and(stream::Output::is_closed)
        {
            let request = self.paused.pop_front().expect("paused request");
            finish_job(request.job, Err(anyhow!("stream client disconnected")));
            return true;
        }
        // Avoid repeatedly restoring a prefix that cannot yet fit with active
        // requests. Live shared prefix pages are accounted for separately.
        if !self.cache.can_fit(&request.tokens) {
            return false;
        }
        let mut request = self.paused.pop_front().expect("paused request");
        match self.attach_for_next(&request.tokens) {
            Ok(Some((sequence, computed))) => {
                request.sequence = sequence;
                request.computed = computed;
                self.active[slot] = Some(request);
                true
            }
            Ok(None) => {
                self.paused.push_front(request);
                false
            }
            Err(error) => {
                finish_job(request.job, Err(error));
                true
            }
        }
    }

    pub(super) fn pause_for_capacity(&mut self) -> Result<()> {
        for slot in 0..self.active.len() {
            if self.active[slot].as_ref().is_some_and(|request| {
                request
                    .job
                    .stream
                    .as_ref()
                    .is_some_and(stream::Output::is_closed)
            }) {
                let request = self.active[slot]
                    .take()
                    .context("missing cancelled request")?;
                self.cache.release(request.sequence)?;
                finish_job(request.job, Err(anyhow!("stream client disconnected")));
            }
        }
        loop {
            let needed = self
                .active
                .iter()
                .flatten()
                .filter(|request| {
                    !request.done && request.computed / 32 >= request.sequence.pages.len()
                })
                .count();
            if needed <= self.cache.available_pages() {
                return Ok(());
            }
            anyhow::ensure!(
                self.active.iter().flatten().count() > 1,
                "one request exceeded its physical KV capacity"
            );
            // Preserve progress toward first tokens before pausing another
            // prefilling request. Larger victims release more private pages.
            let slot = self
                .active
                .iter()
                .enumerate()
                .filter_map(|(slot, request)| request.as_ref().map(|request| (slot, request)))
                .max_by_key(|(slot, request)| {
                    (request.first.is_some(), request.sequence.pages.len(), *slot)
                })
                .context("missing KV preemption victim")?
                .0;
            let mut request = self.active[slot]
                .take()
                .context("missing preemption victim")?;
            self.cache.release(std::mem::take(&mut request.sequence))?;
            request.computed = 0;
            self.paused.push_back(request);
            self.preemptions += 1;
        }
    }
}
