//! Native HiFT source-prefix reuse, eight-frame context and Hamming overlap-add.
use super::*;
impl Engine {
    pub fn new_codec_hift_cache(&self) -> Result<CodecHiftCache> {
        ensure!(
            self.weights.contains_key("hift.conv_post.weight"),
            "HiFT weights are not loaded"
        );
        let window: Vec<f32> = (0..7680)
            .map(|i| (0.54 - 0.46 * (core::f64::consts::TAU * i as f64 / 7679.).cos()) as f32)
            .collect();
        Ok(CodecHiftCache {
            context: self._context.clone(),
            position: 0,
            valid: true,
            finished: false,
            mel: None,
            source: None,
            wave: None,
            window: DeviceBuffer::from_host(&self.stream, &window)?,
        })
    }
    /// Noise spans the retained eight mel frames plus the new frames, in sample/harmonic order.
    pub fn codec_hift_chunk(
        &self,
        cache: &mut CodecHiftCache,
        mel: &[f32],
        noise: &[f32],
        last: bool,
    ) -> Result<Vec<f32>> {
        ensure!(
            Arc::ptr_eq(&cache.context, &self._context) && cache.valid && !cache.finished,
            "invalid, poisoned or finished HiFT cache"
        );
        ensure!(
            !mel.is_empty()
                && mel.len().is_multiple_of(80)
                && mel.iter().chain(noise).all(|v| v.is_finite()),
            "invalid HiFT stream input"
        );
        let frames = mel.len() / 80 + cache.history_frames();
        ensure!(
            frames <= crate::models::minicpm_o45::MAX_MEL_FRAMES
                && noise.len() == frames * 480 * 9
                && (last || frames > 8),
            "invalid HiFT chunk extent or insufficient nonfinal overlap"
        );
        let mel = DeviceBuffer::from_host(&self.stream, mel)?;
        let noise = DeviceBuffer::from_host(&self.stream, noise)?;
        self.codec_hift_chunk_device(cache, &mel, &noise, last)?
            .to_host_vec(&self.stream)
            .map_err(Into::into)
    }
    pub(in crate::inference::codec) fn codec_hift_chunk_device(
        &self,
        cache: &mut CodecHiftCache,
        mel: &DeviceBuffer<f32>,
        noise: &DeviceBuffer<f32>,
        last: bool,
    ) -> Result<Buffer> {
        ensure!(
            Arc::ptr_eq(&cache.context, &self._context) && cache.valid && !cache.finished,
            "invalid HiFT stream cache"
        );
        let frames = cache.history_frames() + mel.len() / 80;
        ensure!(
            !mel.is_empty()
                && mel.len().is_multiple_of(80)
                && noise.len() == frames * 480 * 9
                && frames <= crate::models::minicpm_o45::MAX_MEL_FRAMES
                && (last || frames > 8),
            "invalid device HiFT stream input"
        );
        cache.valid = false;
        let empty = DeviceBuffer::zeroed(&self.stream, 0)?;
        let combined = self.stream_join(cache.mel.as_deref().unwrap_or(&empty), mel, 1, 80)?;
        let f0 = self.hift_f0(&combined, frames)?;
        let source = self.hift_source(&f0, noise)?;
        if let Some(previous) = &cache.source {
            ensure!(
                previous.len() <= source.len(),
                "HiFT source overlap exceeds current source"
            );
            // SAFETY: disjoint saved source replaces a checked prefix of this owned excitation.
            unsafe {
                self.module.codec_stream.stream_prefix(
                    &self.stream,
                    LaunchConfig::for_num_elems(previous.len() as u32),
                    previous,
                    source.cu_deviceptr() as *mut f32,
                )?;
            }
        }
        let spectrum = self.hift_decode(&combined, &source, frames)?;
        let wave = self.hift_inverse(&spectrum)?;
        if let Some(previous) = &cache.wave {
            ensure!(
                previous.len() == 3840 && previous.len() <= wave.len(),
                "invalid HiFT waveform overlap"
            );
            // SAFETY: one bounded previous-wave prefix and a full 2*3840 Hamming window.
            unsafe {
                self.module.codec_stream.stream_blend(
                    &self.stream,
                    LaunchConfig::for_num_elems(previous.len() as u32),
                    previous,
                    &cache.window,
                    wave.cu_deviceptr() as *mut f32,
                )?;
            }
        }
        ensure!(
            source.len() == frames * 480 && wave.len() == source.len(),
            "HiFT stream source/wave extent mismatch"
        );
        let keep = frames.min(8);
        let kept_samples = wave.len().min(3840);
        let next_mel = self.stream_rows(&combined, 1, 80, frames - keep, keep)?;
        let next_source =
            self.stream_rows(&source, 1, 1, source.len() - kept_samples, kept_samples)?;
        let next_wave = self.stream_rows(&wave, 1, 1, wave.len() - kept_samples, kept_samples)?;
        let count = wave.len() - if last { 0 } else { 3840 };
        let output = self.stream_rows(&wave, 1, 1, 0, count)?;
        cache.mel = Some(next_mel);
        cache.source = Some(next_source);
        cache.wave = Some(next_wave);
        cache.position += count;
        cache.finished = last;
        cache.valid = true;
        Ok(output)
    }
}
