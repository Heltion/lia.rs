//! Checked acoustic streaming primitives and per-request encoder/flow state.
use super::*;
mod encoder;
mod flow;
mod flow_block;
mod hift;
mod ops;
mod retention;
mod voice;
pub use voice::CodecVoiceStream;
pub struct CodecEncoderCache {
    context: Arc<CudaContext>,
    position: usize,
    valid: bool,
    finished: bool,
    attention: Vec<Option<Buffer>>,
    lookahead: Buffer,
    upsample: Buffer,
}
impl CodecEncoderCache {
    pub fn history_codes(&self) -> usize {
        self.attention[0].as_ref().map_or(0, |v| v.len() / 1024)
    }
    pub fn history_frames(&self) -> usize {
        self.attention[6].as_ref().map_or(0, |v| v.len() / 1024)
    }
    /// Number of consumed codes; excludes the three unconsumed lookahead codes.
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn finished(&self) -> bool {
        self.finished
    }
}
impl Engine {
    pub fn new_codec_encoder_cache(&self) -> Result<CodecEncoderCache> {
        ensure!(
            self.weights.contains_key("codec.encoder_proj.weight"),
            "codec encoder weights are not loaded"
        );
        Ok(CodecEncoderCache {
            context: self._context.clone(),
            position: 0,
            valid: true,
            finished: false,
            attention: (0..10).map(|_| None).collect(),
            lookahead: self.stream_zeroed(2 * 512)?,
            upsample: self.stream_zeroed(4 * 512)?,
        })
    }
    pub fn codec_encoder_chunk(
        &self,
        cache: &mut CodecEncoderCache,
        codes: &[u32],
        last: bool,
    ) -> Result<Vec<f32>> {
        self.codec_encode_chunk(cache, codes, last)?
            .to_host_vec(&self.stream)
            .map_err(Into::into)
    }
}

/// CFM history belongs to one request, with independent state at every Euler step.
pub struct CodecFlowCache {
    context: Arc<CudaContext>,
    position: usize,
    valid: bool,
    layers: Vec<Vec<FlowLayer>>,
}
struct FlowLayer {
    attention: Option<Buffer>,
    cnn1: Buffer,
    cnn2: Buffer,
}
impl CodecFlowCache {
    pub fn history_frames(&self) -> usize {
        self.layers[0][0]
            .attention
            .as_ref()
            .map_or(0, |v| v.len() / (2 * 1024))
    }
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn steps(&self) -> usize {
        self.layers.len()
    }
}

/// HiFT overlap belongs to one stream and is flushed exactly once.
pub struct CodecHiftCache {
    context: Arc<CudaContext>,
    position: usize,
    valid: bool,
    finished: bool,
    mel: Option<Buffer>,
    source: Option<Buffer>,
    wave: Option<Buffer>,
    window: DeviceBuffer<f32>,
}
impl CodecHiftCache {
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn finished(&self) -> bool {
        self.finished
    }
    pub fn history_frames(&self) -> usize {
        self.mel.as_ref().map_or(0, |v| v.len() / 80)
    }
}
