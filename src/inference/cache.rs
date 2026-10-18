use super::*;

impl Engine {
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn trim_scratch(&self) {
        if self.preallocated {
            return;
        }
        let mut pool = self.pool.borrow_mut();
        let mut available = 64 * 1024 * 1024;
        for (&elements, buffers) in pool.iter_mut().rev() {
            let decode = (1..=8).any(|batch| {
                [1024, 4096, 5120, 6144, 10240, 12288, 17408, 248320]
                    .iter()
                    .any(|&width| elements == batch * width)
            });
            let count = if elements <= 32768 || decode {
                buffers.len().min(available / (elements * 4).max(1))
            } else {
                0
            };
            buffers.truncate(count);
            available -= count * elements * 4;
        }
        pool.retain(|_, buffers| !buffers.is_empty());
    }

    pub fn attention_mma_min_batch(&self) -> usize {
        if self.attention_mma_decode { 1 } else { 16 }
    }

    pub fn fp8_prefill(&self) -> bool {
        self.fp8_work.borrow().is_some()
    }
    pub fn prefill_tile(&self) -> usize {
        self.prefill_tile
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn stochastic_mtp(&self) -> bool {
        self.stochastic_mtp
    }

    pub fn max_draft(&self) -> usize {
        self.max_draft
    }
    pub fn mtp_enabled(&self) -> bool {
        self.geometry.mtp
    }
    pub fn matrix_layout(&self) -> (usize, usize) {
        (self.split_k, self.row_group)
    }
    pub fn matrix_threads(&self) -> usize {
        if self.split_k == 4 {
            128
        } else {
            self.matrix_threads
        }
    }
    pub fn integer_enabled(&self) -> bool {
        self.integer && !self.geometry.moe
    }
    pub fn prefill_precision(&self) -> &'static str {
        if self.geometry.moe {
            "gsq_w2_w8_mixed_f16_f32_accumulation"
        } else if self.fp8_prefill() {
            "mixed_e4m3_f16_f32_accumulation"
        } else {
            "f16"
        }
    }
    pub fn current_token(&self) -> Result<u32> {
        let hidden = self
            .last_hidden
            .as_ref()
            .ok_or_else(|| anyhow!("no target hidden state"))?;
        let logits = self.mm("output.weight", hidden)?;
        Ok(self.greedy(&logits)?[0])
    }
    pub(super) fn copy(&self, input: &DeviceBuffer<f32>) -> Result<DeviceBuffer<f32>> {
        let mut out = DeviceBuffer::zeroed(&self.stream, input.len())?;
        out.copy_from_device_async(input, &self.stream)?;
        Ok(out)
    }
    pub fn refresh_snapshot(&self, saved: &mut Snapshot) -> Result<()> {
        for (state, dest) in self.states.iter().zip(&mut saved.recurrent) {
            if let (State::Recurrent { conv, matrix }, Some((dc, dm))) = (state, dest) {
                dc.copy_from_device_async(conv, &self.stream)?;
                dm.copy_from_device_async(matrix, &self.stream)?;
            }
        }
        match (&self.last_hidden, &mut saved.hidden) {
            (Some(source), Some(dest)) => dest.copy_from_device_async(source, &self.stream)?,
            (Some(source), None) => {
                anyhow::ensure!(
                    !self.preallocated,
                    "checkpoint hidden buffer was not reserved"
                );
                saved.hidden = Some(self.copy(source)?);
            }
            (None, Some(dest)) => dest.zero_async(&self.stream)?,
            (None, None) => {}
        }
        saved.position = self.position;
        saved.positions = self.rotary.borrow().plan.clone();
        saved.images = self.images.clone();
        self.stream.synchronize()?;
        Ok(())
    }
    pub fn snapshot(&self) -> Result<Snapshot> {
        let mut recurrent = Vec::new();
        for state in &self.states {
            recurrent.push(match state {
                State::Recurrent { conv, matrix } => Some((self.copy(conv)?, self.copy(matrix)?)),
                State::Attention { .. } => None,
            });
        }
        let hidden = match &self.last_hidden {
            Some(h) => Some(self.copy(h)?),
            None => None,
        };
        self.stream.synchronize()?;
        Ok(Snapshot {
            position: self.position,
            recurrent,
            hidden,
            positions: self.rotary.borrow().plan.clone(),
            images: self.images.clone(),
        })
    }
    pub fn restore(&mut self, snapshot: &Snapshot) -> Result<()> {
        for (state, saved) in self.states.iter_mut().zip(&snapshot.recurrent) {
            if let (State::Recurrent { conv, matrix }, Some((sc, sm))) = (state, saved) {
                conv.copy_from_device_async(sc, &self.stream)?;
                matrix.copy_from_device_async(sm, &self.stream)?;
            }
        }
        self.last_hidden = match &snapshot.hidden {
            Some(h) => {
                let mut out = self.zeros(h.len())?;
                out.copy_from_device_async(h, &self.stream)?;
                Some(out)
            }
            None => None,
        };
        self.position = snapshot.position;
        self.upload_position_plan(snapshot.positions.clone())?;
        self.images = snapshot.images.clone();
        self.mtp_prepared = None;
        self.stream.synchronize()?;
        Ok(())
    }
    pub fn reset(&mut self) -> Result<()> {
        self.decode_replay.borrow_mut().take();
        self.stream.synchronize()?;
        for state in &mut self.states {
            match state {
                State::Recurrent { conv, matrix } => {
                    conv.zero_async(&self.stream)?;
                    matrix.zero_async(&self.stream)?;
                }
                State::Attention { .. } => {}
            }
        }
        self.position = 0;
        self.rotary.borrow_mut().key = None;
        self.rotary.borrow_mut().plan = None;
        self.images.clear();
        self.verify_hidden = None;
        self.capture = false;
        self.last_hidden = None;
        self.mtp_prepared = None;
        Ok(())
    }
}
