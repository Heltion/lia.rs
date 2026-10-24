//! Independent MiniCPM-o Llama speech-code decoder with its own KV lifetime.
use super::*;
use anyhow::{Context, ensure};
mod chunk;
mod conditioning;
mod generate;
pub use chunk::{DrainedGeneration, StopReason};
pub use generate::Generation;

pub struct Cache {
    context: Arc<CudaContext>,
    capacity: usize,
    position: usize,
    layers: Vec<(DeviceBuffer<f32>, DeviceBuffer<f32>)>,
}
impl Cache {
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn reset(&mut self) {
        self.position = 0;
    }
}

impl Engine {
    pub fn new_speech_cache(&self, capacity: usize) -> Result<Cache> {
        ensure!(
            self.weights.contains_key("speech.head") && (1..=4096).contains(&capacity),
            "invalid speech cache capacity or unloaded decoder"
        );
        let mut layers = Vec::with_capacity(20);
        for layer in 0..20 {
            layers.push((
                DeviceBuffer::zeroed(&self.stream, capacity * 768)
                    .with_context(|| format!("allocating speech key cache {layer}"))?,
                DeviceBuffer::zeroed(&self.stream, capacity * 768)
                    .with_context(|| format!("allocating speech value cache {layer}"))?,
            ));
        }
        Ok(Cache {
            context: self._context.clone(),
            capacity,
            position: 0,
            layers,
        })
    }

    pub fn speech_logits(&self, cache: &mut Cache, embeddings: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            !embeddings.is_empty()
                && embeddings.len().is_multiple_of(768)
                && embeddings.iter().all(|v| v.is_finite()),
            "invalid speech input embeddings"
        );
        let input = DeviceBuffer::from_host(&self.stream, embeddings)?;
        self.speech_forward(cache, &input)?
            .to_host_vec(&self.stream)
            .map_err(Into::into)
    }

    pub fn speech_code_logits(&self, cache: &mut Cache, code: u32) -> Result<Vec<f32>> {
        let embedding = self.speech_embedding("speech.code_embd", &[code])?;
        self.speech_forward(cache, &embedding)?
            .to_host_vec(&self.stream)
            .map_err(Into::into)
    }

    fn speech_forward(&self, cache: &mut Cache, input: &DeviceBuffer<f32>) -> Result<Buffer> {
        let batch = input.len() / 768;
        ensure!(
            Arc::ptr_eq(&cache.context, &self._context)
                && batch > 0
                && batch <= crate::gpu::MAX_PREFILL_BATCH
                && cache.position <= cache.capacity
                && batch <= cache.capacity - cache.position,
            "speech cache context or extent mismatch"
        );
        let mut hidden = self.copy(input)?;
        for layer in 0..20 {
            let prefix = format!("speech.blk.{layer}");
            let normalized = self.norm(&format!("{prefix}.ln1"), &hidden, 768)?;
            let q = self.component_mm(&format!("{prefix}.q"), &normalized, batch, true)?;
            let k = self.component_mm(&format!("{prefix}.k"), &normalized, batch, true)?;
            let v = self.component_mm(&format!("{prefix}.v"), &normalized, batch, true)?;
            let rotated = self.zeros(q.len())?;
            let attended = self.zeros(q.len())?;
            let (kc, vc) = &cache.layers[layer];
            // SAFETY: validated 12 width-64 heads, complete cache rows and disjoint output leases.
            unsafe {
                self.module.speech.speech_rope_kv(
                    &self.stream,
                    LaunchConfig::for_num_elems(q.len() as u32),
                    cache.position,
                    &q,
                    &k,
                    &v,
                    rotated.cu_deviceptr() as *mut f32,
                    kc.cu_deviceptr() as *mut f32,
                    vc.cu_deviceptr() as *mut f32,
                )?;
                self.module.speech.speech_attention(
                    &self.stream,
                    LaunchConfig::for_num_elems((batch * 12 * 32) as u32),
                    cache.position,
                    &rotated,
                    kc,
                    vc,
                    attended.cu_deviceptr() as *mut f32,
                )?;
            }
            let projected = self.component_mm(&format!("{prefix}.out"), &attended, batch, true)?;
            let residual = self.add(&hidden, &projected)?;
            let normalized = self.norm(&format!("{prefix}.ln2"), &residual, 768)?;
            let gate = self.component_mm(&format!("{prefix}.gate"), &normalized, batch, true)?;
            let up = self.component_mm(&format!("{prefix}.up"), &normalized, batch, true)?;
            let activated = self.silu_mul(&up, &gate)?;
            let down = self.component_mm(&format!("{prefix}.down"), &activated, batch, true)?;
            let next = self.add(&residual, &down)?;
            hidden = self.copy(&next)?;
        }
        let normalized = self.norm("speech.norm", &hidden, 768)?;
        let last = self.zeros(768)?;
        // SAFETY: retain the final complete row for the speech-code prediction head.
        unsafe {
            self.module.basic.restore_position(
                &self.stream,
                LaunchConfig::for_num_elems(768),
                (batch - 1) * 768,
                768,
                &normalized,
                last.cu_deviceptr() as *mut f32,
            )?;
        }
        let logits = self.mm("speech.head", &last)?;
        cache.position += batch;
        Ok(logits)
    }
}
