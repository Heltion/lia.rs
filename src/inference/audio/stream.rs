//! Streaming Whisper CNN-context cropping and per-layer bidirectional KV history.
use super::*;

pub struct AudioCache {
    context: Arc<CudaContext>,
    position: usize,
    valid: bool,
    attention: Vec<Option<Buffer>>,
}
impl AudioCache {
    pub fn position(&self) -> usize {
        self.position
    }
}
impl Engine {
    pub fn new_audio_cache(&self) -> Result<AudioCache> {
        ensure!(
            self.audio_input_enabled(),
            "audio encoder weights are not loaded"
        );
        Ok(AudioCache {
            context: self._context.clone(),
            position: 0,
            valid: true,
            attention: (0..24).map(|_| None).collect(),
        })
    }
    pub fn audio_stream_embeddings(
        &self,
        cache: &mut AudioCache,
        features: &Features,
        prefix_frames: usize,
        suffix_frames: usize,
    ) -> Result<Vec<f32>> {
        ensure!(
            cache.valid && Arc::ptr_eq(&cache.context, &self._context),
            "invalid audio stream cache"
        );
        ensure!(
            (1..=3000).contains(&features.frames)
                && features.values.len() == 80 * features.frames
                && features.values.iter().all(|v| v.is_finite()),
            "invalid streaming audio features"
        );
        let original = features.frames.div_ceil(2);
        let prefix = prefix_frames.div_ceil(2);
        let suffix = suffix_frames.div_ceil(2);
        ensure!(
            prefix <= original && suffix <= original - prefix && original - prefix - suffix >= 5,
            "streaming audio CNN crop leaves insufficient pooling rows"
        );
        let frames = original - prefix - suffix;
        // The checkpoint resets against the pre-crop convolution extent, not
        // just the retained core. Learned positions then restart at zero.
        let reset = cache.position + original >= 1500;
        let past = if reset { 0 } else { cache.position };
        ensure!(
            past + frames <= 1500,
            "audio stream exceeds learned position extent"
        );
        let input = DeviceBuffer::from_host(&self.stream, &features.values)?;
        cache.valid = false;
        let first = self.audio_conv::<true>(&input, features.frames, 80, 1, "conv1")?;
        let second = self.audio_conv::<false>(&first, features.frames, 1024, 2, "conv2")?;
        let mut hidden = self.zeros(frames * 1024)?;
        // SAFETY: checked CNN crop and learned-position interval, complete rows.
        unsafe {
            self.module.whisper.whisper_stream_position(
                &self.stream,
                LaunchConfig::for_num_elems(hidden.len() as u32),
                prefix,
                past,
                &second,
                self.floats("audio.position")?,
                &mut hidden,
            )?;
        }
        if reset {
            cache.attention.iter_mut().for_each(|v| *v = None);
        }
        for layer in 0..24 {
            let name = format!("blk.{layer}");
            let normalized = self.audio_norm(&format!("{name}.ln1"), &hidden)?;
            let qkv = self.component_mm(
                &format!("audio.{name}.qkv.weight"),
                &normalized,
                frames,
                true,
            )?;
            let empty = DeviceBuffer::zeroed(&self.stream, 0)?;
            let old = cache.attention[layer].as_deref().unwrap_or(&empty);
            ensure!(old.len() == past * 2048, "inconsistent audio layer cache");
            let mut kv = self.cache_buffer((past + frames) * 2048)?;
            let attention = self.zeros(frames * 1024)?;
            // SAFETY: isolated old/new cache spans, Q/V bias vectors, and one
            // warp per complete width-64 head; all current rows see full history.
            unsafe {
                self.module.whisper.whisper_cache(
                    &self.stream,
                    LaunchConfig::for_num_elems(kv.len() as u32),
                    past,
                    old,
                    &qkv,
                    self.floats(&format!("audio.{name}.v.bias"))?,
                    &mut kv,
                )?;
                self.module.whisper.whisper_cached_attention(
                    &self.stream,
                    LaunchConfig::for_num_elems((frames * 16 * 32) as u32),
                    frames,
                    past + frames,
                    &qkv,
                    &kv,
                    self.floats(&format!("audio.{name}.q.bias"))?,
                    attention.cu_deviceptr() as *mut f32,
                )?;
            }
            hidden = self.audio_residual_block(&name, &hidden, &attention, frames)?;
            cache.attention[layer] = Some(kv);
        }
        let encoded = self.audio_norm("post_ln", &hidden)?;
        let projected = self.component_mm("audio.proj1.weight", &encoded, frames, true)?;
        let projected = self.audio_affine(projected, "proj1.bias", Activation::Relu)?;
        let projected = self.component_mm("audio.proj2.weight", &projected, frames, true)?;
        let projected = self.audio_affine(projected, "proj2.bias", Activation::Linear)?;
        let mut output = self.zeros(frames / 5 * 4096)?;
        // SAFETY: only complete five-row groups are pooled to 4096-wide rows.
        unsafe {
            self.module.whisper.whisper_pool(
                &self.stream,
                LaunchConfig::for_num_elems(output.len() as u32),
                &projected,
                &mut output,
            )?;
        }
        let output = output.to_host_vec(&self.stream)?;
        cache.position = past + frames;
        cache.valid = true;
        self.trim_scratch();
        Ok(output)
    }
}
