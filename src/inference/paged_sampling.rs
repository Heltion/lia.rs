//! Sparse sampler readback retains exact full-vocabulary nucleus semantics.
use super::*;
use crate::sampling::Sampling;
use anyhow::{Context, ensure};

impl Engine {
    pub fn forward_paged_sampled(
        &mut self,
        rows: &[PagedRow],
        tables: &[Vec<u32>],
        outputs: &[u32],
        samplers: &mut [Option<Sampling>],
    ) -> Result<Vec<u32>> {
        ensure!(
            samplers.len() == outputs.len(),
            "sampler output count mismatch"
        );
        let temperatures: Vec<_> = samplers
            .iter()
            .map(|s| {
                s.as_ref().map_or(0.0, |s| {
                    let (temperature, top_k, _) = s.settings();
                    if top_k == 0 { temperature } else { 0.0 }
                })
            })
            .collect();
        for (&output, sampler) in outputs.iter().zip(samplers.iter_mut()) {
            let row = rows
                .get(output as usize)
                .context("sampler row outside packed input")?;
            if let Some(sampler) = sampler {
                sampler.set_position(u64::from(row.position));
            }
        }
        self.nucleus
            .borrow_mut()
            .as_mut()
            .context("paged sampler is not configured")?
            .prepare(&self.stream, &temperatures)?;
        self.run_paged(rows, tables, outputs, |engine, logits| {
            let Some(logits) = logits else {
                return Ok(Vec::new());
            };
            let mut work = engine.nucleus.borrow_mut();
            let work = work.as_mut().context("missing sampler workspace")?;
            let candidates = work.read(&engine.module, &engine.stream, logits)?;
            let mut result = Vec::new();
            for (index, (row, sampler)) in candidates.into_iter().zip(samplers).enumerate() {
                let token = if let Some(sampler) = sampler {
                    let distribution = if sampler.settings().1 != 0 {
                        sampler.distribution(row.candidates)?
                    } else if let Some(distribution) = sampler.checked_nucleus(
                        row.candidates,
                        row.maximum,
                        row.total,
                        row.omitted,
                    )? {
                        distribution
                    } else {
                        work.fallbacks += 1;
                        sampler.distribution_logits(&engine.read_logit_row(logits, index)?)?
                    };
                    sampler.draw(&distribution)
                } else {
                    row.candidates
                        .iter()
                        .max_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
                        .context("empty greedy vocabulary")?
                        .0
                };
                result.push(token);
            }
            Ok(result)
        })
    }

    fn read_logit_row(&self, logits: &DeviceBuffer<f32>, row: usize) -> Result<Vec<f32>> {
        let vocabulary = self.vocabulary_size();
        ensure!(
            (row + 1) * vocabulary <= logits.len(),
            "logit row outside GPU output"
        );
        let mut output = vec![0_f32; vocabulary];
        // SAFETY: validated complete row, owned host destination retained until synchronization.
        let status = unsafe {
            cuda_core::sys::cuMemcpyDtoHAsync_v2(
                output.as_mut_ptr().cast(),
                logits.cu_deviceptr() + (row * vocabulary * 4) as u64,
                vocabulary * 4,
                self.stream.cu_stream(),
            )
        };
        ensure!(
            status == 0,
            "reading full sampler fallback: {:?}",
            cuda_core::DriverError(status)
        );
        self.stream.synchronize()?;
        Ok(output)
    }

    pub fn nucleus_fallbacks(&self) -> u64 {
        self.nucleus.borrow().as_ref().map_or(0, |w| w.fallbacks)
    }
}
