//! Request-owned acoustic synthesis with reference priming and bounded history.
use super::*;
use crate::models::minicpm_o45::Voice;

pub struct CodecVoiceStream {
    encoder: CodecEncoderCache,
    flow: CodecFlowCache,
    hift: CodecHiftCache,
    speaker: Buffer,
    prefix: usize,
    valid: bool,
}
impl CodecVoiceStream {
    pub fn finished(&self) -> bool {
        self.hift.finished()
    }
    pub fn history_frames(&self) -> usize {
        self.flow.history_frames()
    }
    pub fn overlap_frames(&self) -> usize {
        self.hift.history_frames()
    }
    pub fn generated_samples(&self) -> usize {
        self.hift.position()
    }
}
impl Engine {
    /// Prime the acoustic caches with reference codes and mel. Noise is supplied
    /// explicitly so callers own reproducibility independently of token sampling.
    pub fn new_codec_voice_stream(
        &self,
        voice: &Voice<'_>,
        steps: usize,
        noise: &[f32],
    ) -> Result<CodecVoiceStream> {
        ensure!(
            (voice.codes().is_empty() || voice.codes().len() >= 2)
                && noise.len() == voice.mel().len()
                && noise.iter().all(|v| v.is_finite()),
            "invalid streaming reference extent or noise"
        );
        let mut encoder = self.new_codec_encoder_cache()?;
        let mut flow = self.new_codec_flow_cache(steps)?;
        let hift = self.new_codec_hift_cache()?;
        let norm = (voice
            .embedding()
            .iter()
            .map(|v| f64::from(*v).powi(2))
            .sum::<f64>()
            .sqrt() as f32)
            .max(1e-12);
        let embedding: Vec<f32> = voice.embedding().iter().map(|v| v / norm).collect();
        let embedding = DeviceBuffer::from_host(&self.stream, &embedding)?;
        let speaker =
            self.codec_linear("spk_embed_affine_layer", &embedding, 1, Activation::Linear)?;
        let prefix = voice.mel().len() / 80;
        if !voice.codes().is_empty() {
            let mut codes = voice.codes().to_vec();
            codes.extend(voice.codes().iter().copied().cycle().take(3));
            let mu = self.codec_encode_chunk(&mut encoder, &codes, false)?;
            let noise = DeviceBuffer::from_host(&self.stream, noise)?;
            let condition = DeviceBuffer::from_host(&self.stream, voice.mel())?;
            self.codec_flow_chunk_device(&mut flow, &noise, &mu, &speaker, &condition)?;
        }
        self.trim_scratch();
        Ok(CodecVoiceStream {
            encoder,
            flow,
            hift,
            speaker,
            prefix,
            valid: true,
        })
    }

    /// Nonfinal code slices include three unconsumed lookahead codes. The final
    /// slice consumes every code and flushes the waveform overlap exactly once.
    pub fn codec_voice_stream_chunk(
        &self,
        cache: &mut CodecVoiceStream,
        codes: &[u32],
        flow_noise: &[f32],
        hift_noise: &[f32],
        last: bool,
    ) -> Result<Vec<f32>> {
        ensure!(
            cache.valid && !cache.finished() && Arc::ptr_eq(&cache.encoder.context, &self._context),
            "invalid or finished acoustic stream"
        );
        ensure!(
            !codes.is_empty() && codes.iter().all(|&v| v < 6561) && (last || codes.len() >= 5),
            "invalid acoustic code chunk"
        );
        let consumed = if last { codes.len() } else { codes.len() - 3 };
        let frames = consumed * 2;
        ensure!(
            flow_noise.len() == frames * 80
                && hift_noise.len() == (frames + cache.overlap_frames()) * 480 * 9
                && flow_noise.iter().chain(hift_noise).all(|v| v.is_finite()),
            "invalid acoustic chunk noise"
        );
        let flow_noise = DeviceBuffer::from_host(&self.stream, flow_noise)?;
        let hift_noise = DeviceBuffer::from_host(&self.stream, hift_noise)?;
        cache.valid = false;
        let mu = self.codec_encode_chunk(&mut cache.encoder, codes, last)?;
        let condition = self.stream_zeroed(frames * 80)?;
        let mel = self.codec_flow_chunk_device(
            &mut cache.flow,
            &flow_noise,
            &mu,
            &cache.speaker,
            &condition,
        )?;
        let wave = self.codec_hift_chunk_device(&mut cache.hift, &mel, &hift_noise, last)?;
        self.trim_codec_encoder_cache(&mut cache.encoder, cache.prefix, 100)?;
        self.trim_codec_flow_cache(&mut cache.flow, cache.prefix, 100)?;
        let wave = wave.to_host_vec(&self.stream)?;
        cache.valid = true;
        self.trim_scratch();
        Ok(wave)
    }
}
