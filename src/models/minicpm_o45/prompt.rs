//! Expand image placeholders into 64-query slice spans with ordinary text positions.
use crate::{
    tokenizer::Tokenizer,
    vision::{PositionPlan, PreparedImage},
};
use anyhow::{Result, ensure};

pub struct PreparedPrompt {
    pub tokens: Vec<u32>,
    pub images: Vec<(usize, PreparedImage)>,
    pub positions: PositionPlan,
    pub audios: Vec<(usize, super::audio::Features)>,
}

pub fn prepare_prompt(
    tokenizer: &Tokenizer,
    tokens: Vec<u32>,
    sources: &[String],
    max_slices: usize,
    capacity: usize,
) -> Result<PreparedPrompt> {
    prepare_media_prompt(tokenizer, tokens, sources, &[], max_slices, capacity)
}

pub fn prepare_media_prompt(
    tokenizer: &Tokenizer,
    tokens: Vec<u32>,
    sources: &[String],
    audio_sources: &[String],
    max_slices: usize,
    capacity: usize,
) -> Result<PreparedPrompt> {
    let single = |text: &str| -> Result<u32> {
        let ids = tokenizer.encode(text, true)?;
        ensure!(ids.len() == 1, "expected one special token for {text}");
        Ok(ids[0])
    };
    let marker = single("<|image_pad|>")?;
    let unknown = single("<unk>")?;
    let audio_marker = single("<|audio|>")?;
    ensure!(
        tokens.iter().filter(|&&id| id == audio_marker).count() == audio_sources.len(),
        "audio markers do not match inputs"
    );
    ensure!(
        tokens.iter().filter(|&&id| id == marker).count() == sources.len(),
        "image markers do not match inputs"
    );
    let mut expanded = Vec::new();
    let mut images = Vec::new();
    let mut index = 0;
    let mut audios = Vec::new();
    let mut audio_index = 0;
    for token in tokens {
        if token == audio_marker {
            let prepared = super::audio_prompt::prepare(tokenizer, &audio_sources[audio_index])?;
            for (start, features) in prepared.spans {
                audios.push((expanded.len() + start, features));
            }
            expanded.extend(prepared.tokens);
            audio_index += 1;
            continue;
        }
        if token != marker {
            expanded.push(token);
            continue;
        }
        let bytes = crate::vision::data_url_bytes(&sources[index])?;
        let prepared = super::prepare(&bytes, max_slices)?;
        let replacement = tokenizer.encode(&prepared.plan.placeholder(index, true), true)?;
        let mut ranges = Vec::new();
        let mut cursor = 0;
        while cursor < replacement.len() {
            if replacement[cursor] != unknown {
                cursor += 1;
                continue;
            }
            let start = cursor;
            while cursor < replacement.len() && replacement[cursor] == unknown {
                cursor += 1;
            }
            ensure!(
                cursor - start == 64,
                "resampler span must contain exactly 64 tokens"
            );
            ranges.push(expanded.len() + start);
        }
        ensure!(
            ranges.len() == prepared.slices.len(),
            "image slices and resampler spans differ"
        );
        for (start, image) in ranges.into_iter().zip(prepared.slices) {
            images.push((start, image));
        }
        expanded.extend(replacement);
        index += 1;
    }
    ensure!(
        expanded.len() <= capacity,
        "image prompt exceeds context capacity"
    );
    let mut positions = PositionPlan::new();
    positions.append_text(expanded.len())?;
    Ok(PreparedPrompt {
        tokens: expanded,
        images,
        positions,
        audios,
    })
}
