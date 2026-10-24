//! Sparse readback is safe only with a complete softmax normalizer and omitted-tail bound.
use super::*;
use anyhow::{Result, ensure};
const SLOTS: usize = 21;

pub(crate) struct Row {
    pub candidates: Vec<(u32, f32)>,
    pub maximum: f64,
    pub total: f64,
    pub omitted: f32,
}

pub(crate) struct Workspace {
    vocabulary: usize,
    parts: usize,
    candidates: DeviceBuffer<u64>,
    normalizers: DeviceBuffer<f64>,
    temperatures: DeviceBuffer<f64>,
    staged: Vec<f64>,
    pub fallbacks: u64,
}

impl Workspace {
    pub fn new(stream: &CudaStream, vocabulary: usize, rows: usize) -> Result<Self> {
        let parts = vocabulary.div_ceil(1024);
        Ok(Self {
            vocabulary,
            parts,
            candidates: DeviceBuffer::zeroed(stream, rows * parts * SLOTS)?,
            normalizers: DeviceBuffer::zeroed(stream, rows * parts * 2)?,
            temperatures: DeviceBuffer::zeroed(stream, rows)?,
            staged: Vec::new(),
            fallbacks: 0,
        })
    }
    pub fn prepare(&mut self, stream: &CudaStream, temperatures: &[f64]) -> Result<()> {
        ensure!(
            temperatures.len() <= self.temperatures.len(),
            "sampler batch exceeds reservation"
        );
        if self.staged == temperatures {
            return Ok(());
        }
        // SAFETY: validated reserved extent and borrowed source retained until synchronization.
        let status = unsafe {
            cuda_core::sys::cuMemcpyHtoDAsync_v2(
                self.temperatures.cu_deviceptr(),
                temperatures.as_ptr().cast(),
                std::mem::size_of_val(temperatures),
                stream.cu_stream(),
            )
        };
        ensure!(
            status == 0,
            "staging sampler temperatures: {:?}",
            cuda_core::DriverError(status)
        );
        stream.synchronize()?;
        self.staged = temperatures.to_vec();
        Ok(())
    }
    pub fn read(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        logits: &DeviceBuffer<f32>,
    ) -> Result<Vec<Row>> {
        ensure!(
            logits.len().is_multiple_of(self.vocabulary),
            "invalid sampler logits"
        );
        let rows = logits.len() / self.vocabulary;
        ensure!(
            rows == self.staged.len(),
            "sampler temperature shape mismatch"
        );
        // SAFETY: complete vocabulary rows, reserved partition records and full warps.
        unsafe {
            module.llama_nucleus.llama_nucleus_candidates(
                stream,
                LaunchConfig::for_num_elems((rows * self.parts * 32) as u32),
                self.vocabulary,
                self.parts,
                SLOTS,
                &self.temperatures,
                logits,
                self.candidates.cu_deviceptr() as *mut u64,
                self.normalizers.cu_deviceptr() as *mut f64,
            )?;
        }
        let candidates = self.candidates.to_host_vec(stream)?;
        let normalizers = self.normalizers.to_host_vec(stream)?;
        let mut result = Vec::new();
        for row in 0..rows {
            let norms = &normalizers[row * self.parts * 2..(row + 1) * self.parts * 2];
            ensure!(
                norms.chunks_exact(2).all(|p| p[1].is_finite()),
                "invalid sampler logits or normalizer"
            );
            let maximum = norms
                .chunks_exact(2)
                .map(|p| p[0])
                .max_by(f64::total_cmp)
                .unwrap_or(f64::NEG_INFINITY);
            let temperature = self.staged[row];
            let total = if temperature > 0.0 {
                norms
                    .chunks_exact(2)
                    .filter(|p| p[1] > 0.0)
                    .map(|p| p[1] * ((p[0] - maximum) / temperature).exp())
                    .sum()
            } else {
                0.0
            };
            let mut selected = Vec::new();
            let mut omitted = f32::NEG_INFINITY;
            for part in 0..self.parts {
                let base = (row * self.parts + part) * SLOTS;
                for (rank, &word) in candidates[base..base + SLOTS].iter().enumerate() {
                    let id = (word >> 32) as u32;
                    let value = f32::from_bits(word as u32);
                    if id == u32::MAX || !value.is_finite() {
                        continue;
                    }
                    ensure!(
                        (id as usize) < self.vocabulary,
                        "sampler candidate ID outside vocabulary"
                    );
                    if rank < SLOTS - 1 {
                        selected.push((id, value));
                    } else {
                        omitted = omitted.max(value);
                    }
                }
            }
            result.push(Row {
                candidates: selected,
                maximum,
                total,
                omitted,
            });
        }
        Ok(result)
    }
}
