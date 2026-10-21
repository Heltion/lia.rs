use super::canvas::{self, ImageSpan, Markers};
use crate::{
    tokenizer::Tokenizer,
    vision::{PositionPlan, PreparedImage},
};
use anyhow::{Context, Result, ensure};

pub struct PreparedPrompt {
    pub tokens: Vec<u32>,
    pub images: Vec<(usize, PreparedImage)>,
    pub positions: PositionPlan,
}

// Image placeholders preserve local image IDs and row-major slice boundaries.
use super::images::ImagePlan;

impl ImagePlan {
    pub fn placeholder(&self, image_index: usize) -> String {
        let sizes = self.slice_dimensions();
        let tokens = |[w, h]: [usize; 2]| "<|image_pad|>".repeat(w / 56 * (h / 56));
        let mut text = format!(
            "<image_id>{image_index}</image_id><image>{}</image>",
            tokens(sizes[0])
        );
        if let Some([columns, rows]) = self.grid {
            let slice = format!("<slice>{}</slice>", tokens(sizes[1]));
            for row in 0..rows {
                if row > 0 {
                    text.push('\n');
                }
                for _ in 0..columns {
                    text.push_str(&slice);
                }
            }
        }
        text
    }
}

pub fn prepare_prompt(
    tokenizer: &Tokenizer,
    tokens: Vec<u32>,
    sources: &[String],
    max_slices: usize,
    capacity: usize,
) -> Result<PreparedPrompt> {
    let single = |text: &str| -> Result<u32> {
        let ids = tokenizer.encode(text, true)?;
        ensure!(ids.len() == 1, "MiniCPM marker must be one token: {text}");
        Ok(ids[0])
    };
    let marker = single("<|image_pad|>")?;
    ensure!(
        tokens.iter().filter(|&&id| id == marker).count() == sources.len(),
        "rendered image markers do not match image inputs"
    );
    let mut expanded = Vec::new();
    let mut images = Vec::new();
    let mut spans = Vec::new();
    let mut image_index = 0;
    for token in tokens {
        if token != marker {
            expanded.push(token);
            continue;
        }
        let bytes = crate::vision::data_url_bytes(&sources[image_index])?;
        let prepared = super::prepare(&bytes, max_slices)?;
        let replacement = tokenizer.encode(&prepared.plan.placeholder(image_index), true)?;
        let offset = expanded.len();
        let mut ranges = Vec::new();
        let mut i = 0;
        while i < replacement.len() {
            if replacement[i] != marker {
                i += 1;
                continue;
            }
            let begin = i;
            while i < replacement.len() && replacement[i] == marker {
                i += 1;
            }
            ranges.push(offset + begin..offset + i);
        }
        ensure!(
            ranges.len() == prepared.slices.len(),
            "MiniCPM visual slice span mismatch"
        );
        for (image, range) in prepared.slices.into_iter().zip(&ranges) {
            ensure!(
                range.len() == image.width / 56 * (image.height / 56),
                "MiniCPM visual token count mismatch"
            );
            images.push((range.start, image));
        }
        spans.push(ImageSpan {
            plan: prepared.plan,
            ranges,
        });
        expanded.extend(replacement);
        image_index += 1;
    }
    ensure!(
        expanded.len() <= capacity,
        "image prompt exceeds context capacity"
    );
    let markers = Markers {
        image_start: single("<image>")?,
        image_end: single("</image>")?,
        slice_start: single("<slice>")?,
        slice_end: single("</slice>")?,
        newline: single("\n")?,
    };
    let coordinates = canvas::positions(&expanded, &spans, markers)
        .context("building MiniCPM canvas positions")?;
    let positions = PositionPlan::from_coordinates(coordinates.positions, coordinates.next_text)?;
    Ok(PreparedPrompt {
        tokens: expanded,
        images,
        positions,
    })
}
