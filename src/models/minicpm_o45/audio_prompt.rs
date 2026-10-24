//! Ordinary audio placeholders and chunk features follow the checkpoint processor.
use super::audio::{Features, Processor};
use crate::tokenizer::Tokenizer;
use anyhow::{Context, Result, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};

pub(super) struct Expansion {
    pub tokens: Vec<u32>,
    pub spans: Vec<(usize, Features)>,
}

pub(super) fn prepare(tokenizer: &Tokenizer, source: &str) -> Result<Expansion> {
    let (header, data) = source
        .split_once(',')
        .context("missing audio data URL payload")?;
    ensure!(
        [
            "data:audio/wav;base64",
            "data:audio/x-wav;base64",
            "data:audio/wave;base64"
        ]
        .contains(&header),
        "expected WAV base64 data URL"
    );
    ensure!(
        data.len() <= 128 * 1024 * 1024,
        "audio data URL exceeds limit"
    );
    let bytes = STANDARD.decode(data).context("decoding audio base64")?;
    let audio = super::wav::decode(&bytes)?;
    ensure!(!audio.samples.is_empty(), "empty audio input");
    let samples = if audio.sample_rate == 16000 {
        audio.samples
    } else {
        super::resample::Resampler::new(audio.sample_rate, 16000)?.process(&audio.samples)?
    };
    let processor = Processor::default();
    let begin = tokenizer.encode("<|audio_start|>", true)?;
    let end = tokenizer.encode("<|audio_end|>", true)?;
    let unknown = tokenizer.encode("<unk>", true)?;
    ensure!(
        begin.len() == 1 && end.len() == 1 && unknown.len() == 1,
        "invalid audio special tokens"
    );
    let mut tokens = begin;
    let mut spans = Vec::new();
    for chunk in samples.chunks(480_000) {
        let features = processor.extract_padded(chunk)?;
        let count = features.frames.div_ceil(2) / 5;
        ensure!(
            count > 0,
            "audio chunk is shorter than one encoder pooling group"
        );
        spans.push((tokens.len(), features));
        tokens.extend(std::iter::repeat_n(unknown[0], count));
    }
    tokens.extend(end);
    Ok(Expansion { tokens, spans })
}
