//! Synthesize and deliver confirmed speech codes before decoder EOS.
use super::*;
use crate::inference::CodecVoiceStream;

pub(super) struct Render {
    cache: CodecVoiceStream,
    pending: Vec<u32>,
    seed: u64,
    chunk: u64,
    confirmed: usize,
    sample_offset: usize,
    pub wave: Vec<f32>,
    pub first_audio_ms: Option<f64>,
    pub first_audio_code_n: Option<usize>,
    pub waveform_ms: f64,
}
impl Render {
    pub fn new(
        engine: &Engine,
        voice: &crate::models::minicpm_o45::Voice<'_>,
        seed: u64,
        sample_offset: usize,
    ) -> Result<Self> {
        let start = Instant::now();
        let noise = crate::sampling::gaussian(seed, voice.mel().len());
        let cache = engine.new_codec_voice_stream(voice, 10, &noise)?;
        Ok(Self {
            cache,
            pending: Vec::new(),
            seed,
            chunk: 1,
            confirmed: 0,
            sample_offset,
            wave: Vec::new(),
            first_audio_ms: None,
            first_audio_code_n: None,
            waveform_ms: start.elapsed().as_secs_f64() * 1000.,
        })
    }
    pub fn push(
        &mut self,
        engine: &Engine,
        code: u32,
        output: &stream::Output,
        format: &str,
        started: Instant,
    ) -> Result<()> {
        self.pending.push(code);
        self.confirmed += 1;
        if self.pending.len() == 28 {
            self.emit(engine, output, format, started, false)?;
            self.pending.drain(..25);
        }
        Ok(())
    }
    pub fn finish(
        &mut self,
        engine: &Engine,
        output: &stream::Output,
        format: &str,
        started: Instant,
    ) -> Result<()> {
        anyhow::ensure!(
            !self.pending.is_empty(),
            "speech stream has no codes to flush"
        );
        self.emit(engine, output, format, started, true)?;
        self.pending.clear();
        Ok(())
    }
    fn emit(
        &mut self,
        engine: &Engine,
        output: &stream::Output,
        format: &str,
        started: Instant,
        last: bool,
    ) -> Result<()> {
        anyhow::ensure!(
            !output.is_closed(),
            "stream client disconnected during waveform synthesis"
        );
        let start = Instant::now();
        let frames = (self.pending.len() - if last { 0 } else { 3 }) * 2;
        let flow = crate::sampling::gaussian(self.seed.wrapping_add(self.chunk), frames * 80);
        let hift = crate::sampling::gaussian(
            self.seed.wrapping_add(self.chunk) ^ 0x9e37_79b9_7f4a_7c15,
            (frames + self.cache.overlap_frames()) * 480 * 9,
        );
        let wave =
            engine.codec_voice_stream_chunk(&mut self.cache, &self.pending, &flow, &hift, last)?;
        self.waveform_ms += start.elapsed().as_secs_f64() * 1000.;
        let wav = crate::models::minicpm_o45::wav::pcm16(&wave, 24000)?;
        let bytes = if format == "pcm16" { &wav[44..] } else { &wav };
        output.audio(
            bytes,
            format,
            self.sample_offset + self.wave.len(),
            self.confirmed,
            last,
        )?;
        if self.first_audio_ms.is_none() {
            self.first_audio_ms = Some(started.elapsed().as_secs_f64() * 1000.);
            self.first_audio_code_n = Some(self.confirmed);
        }
        self.wave.extend(wave);
        self.chunk += 1;
        Ok(())
    }
}
