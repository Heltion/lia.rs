use super::*;
use crate::sampling::{Distribution, Sampling};

impl Engine {
    pub fn current_sample(&self, sampling: &mut Sampling) -> Result<u32> {
        let hidden = self
            .last_hidden
            .as_ref()
            .ok_or_else(|| anyhow!("no target hidden state"))?;
        let logits = self.mm("output.weight", hidden)?;
        let distributions = self.distributions(&logits, sampling)?;
        Ok(sampling.draw(
            distributions
                .first()
                .ok_or_else(|| anyhow!("missing target distribution"))?,
        ))
    }

    fn distributions(
        &self,
        logits: &DeviceBuffer<f32>,
        sampling: &Sampling,
    ) -> Result<Vec<Distribution>> {
        const VOCAB: usize = 248320;
        anyhow::ensure!(
            !logits.is_empty() && logits.len().is_multiple_of(VOCAB),
            "invalid sampling logit shape"
        );
        let batch = logits.len() / VOCAB;
        let parts = VOCAB.div_ceil(1024);
        let slots = parts * 20;
        let required = batch * slots;
        let mut workspace = self.sampling_work.borrow_mut();
        if let std::collections::btree_map::Entry::Vacant(entry) = workspace.entry(batch) {
            entry.insert(SelectionWork {
                candidates: DeviceBuffer::zeroed(&self.stream, required)?,
            });
        }
        let workspace = workspace
            .get(&batch)
            .ok_or_else(|| anyhow!("missing sampling workspace"))?;
        // SAFETY: complete vocabulary rows, full warp partitions, and20 output
        // slots per partition. Workspaces remain live on the single stream.
        unsafe {
            self.module.topk.top20(
                &self.stream,
                LaunchConfig::for_num_elems((batch * parts * 32) as u32),
                VOCAB,
                parts,
                logits,
                workspace.candidates.cu_deviceptr() as *mut u64,
            )?;
        }
        let candidates = workspace.candidates.to_host_vec(&self.stream)?;
        candidates
            .chunks(slots)
            .map(|candidates| {
                sampling.distribution(
                    candidates
                        .iter()
                        .map(|word| ((word >> 32) as u32, f32::from_bits(*word as u32)))
                        .filter(|(id, _)| *id != u32::MAX),
                )
            })
            .collect()
    }

    pub(super) fn select_verified(
        &self,
        logits: &DeviceBuffer<f32>,
        drafts: &[u32],
        eos: u32,
        sampling: Option<&mut Sampling>,
    ) -> Result<(usize, u32)> {
        if let Some(sampling) = sampling {
            let distributions = self.distributions(logits, sampling)?;
            return sampling.select_prefix(&distributions, drafts, eos);
        }
        let verified = self.greedy(logits)?;
        let mut accepted = 0;
        for (index, proposal) in drafts.iter().enumerate() {
            if verified[index] != *proposal {
                break;
            }
            accepted += 1;
            if *proposal == eos {
                break;
            }
        }
        Ok((accepted, verified[accepted]))
    }
}
