//! Three-axis positions for an image thumbnail and its spatial slice canvas.
use super::images::ImagePlan;
use anyhow::{Result, ensure};
use std::ops::Range;

pub struct ImageSpan {
    pub plan: ImagePlan,
    /// Visual token spans, excluding image/slice delimiters, in prompt order.
    pub ranges: Vec<Range<usize>>,
}

#[derive(Clone, Copy)]
pub struct Markers {
    pub image_start: u32,
    pub image_end: u32,
    pub slice_start: u32,
    pub slice_end: u32,
    pub newline: u32,
}

impl Markers {
    fn structural(self, token: u32) -> bool {
        [
            self.image_start,
            self.image_end,
            self.slice_start,
            self.slice_end,
            self.newline,
        ]
        .contains(&token)
    }
}

pub struct Coordinates {
    pub positions: Vec<[u32; 4]>,
    pub next_text: u32,
}

fn thumbnail_coordinate(index: usize, size: usize, extent: usize) -> u32 {
    if size <= 1 {
        return 0;
    }
    let end = (extent - 1) as f32;
    let step = end / (size - 1) as f32;
    let value = if index < size / 2 {
        step * index as f32
    } else {
        end - step * (size - index - 1) as f32
    };
    value.round_ties_even() as u32
}

pub fn positions(tokens: &[u32], images: &[ImageSpan], markers: Markers) -> Result<Coordinates> {
    ensure!(
        tokens.len() <= crate::MAX_CONTEXT_TOKENS,
        "image prompt exceeds position limit"
    );
    let mut output = Vec::with_capacity(tokens.len());
    let mut base = 0u32;
    for image in images {
        let sizes = image.plan.slice_dimensions();
        ensure!(
            sizes.len() == image.ranges.len() && !image.ranges.is_empty(),
            "canvas slice count mismatch"
        );
        for (i, (range, [w, h])) in image.ranges.iter().zip(&sizes).enumerate() {
            ensure!(
                range.start > 0
                    && range.end < tokens.len()
                    && range.start < range.end
                    && range.end - range.start == w / 56 * (h / 56),
                "invalid canvas visual span"
            );
            let (start, end) = if i == 0 {
                (markers.image_start, markers.image_end)
            } else {
                (markers.slice_start, markers.slice_end)
            };
            ensure!(
                tokens[range.start - 1] == start && tokens[range.end] == end,
                "canvas delimiter mismatch"
            );
            if i > 0 {
                ensure!(
                    image.ranges[i - 1].end < range.start,
                    "overlapping canvas visual spans"
                );
            }
        }
        let start = image.ranges[0].start - 1;
        ensure!(
            start >= output.len(),
            "canvas images are not in prompt order"
        );
        while output.len() < start {
            output.push([base; 4]);
            base += 1;
        }
        let mut end = image.ranges.last().unwrap().end;
        while end < tokens.len() && markers.structural(tokens[end]) {
            end += 1;
        }
        let [thumb_w, thumb_h] = sizes[0].map(|x| x / 56);
        let (width, height) = if let Some([cols, rows]) = image.plan.grid {
            (sizes[1][0] / 56 * cols, sizes[1][1] / 56 * rows)
        } else {
            (thumb_w, thumb_h)
        };
        ensure!(width > 0 && height > 0, "empty canvas");
        let next = base
            .checked_add(width.max(height) as u32 + 1)
            .ok_or_else(|| anyhow::anyhow!("canvas position overflow"))?;
        output.resize(end, [base, base, base, 0]);
        output[start] = [base, base.saturating_sub(1), base.saturating_sub(1), 0];
        let thumbnail = &image.ranges[0];
        for (i, at) in (thumbnail.start..thumbnail.end).enumerate() {
            let y = i / thumb_w;
            let x = i % thumb_w;
            let (y, x) = if image.plan.grid.is_some() {
                (
                    thumbnail_coordinate(y, thumb_h, height),
                    thumbnail_coordinate(x, thumb_w, width),
                )
            } else {
                (y as u32, x as u32)
            };
            output[at] = [base, base + y, base + x, 0];
        }
        output[thumbnail.end] = [base, base + height as u32, base + width as u32, 0];
        if let Some([columns, _]) = image.plan.grid {
            let [slice_w, slice_h] = sizes[1].map(|x| x / 56);
            for (i, range) in image.ranges[1..].iter().enumerate() {
                let y = i / columns * slice_h;
                let x = i % columns * slice_w;
                output[range.start - 1] = [base, base + y as u32, base + x as u32, 0];
                for (j, at) in (range.start..range.end).enumerate() {
                    output[at] = [
                        base,
                        base + (y + j / slice_w) as u32,
                        base + (x + j % slice_w) as u32,
                        0,
                    ];
                }
                output[range.end] = [
                    base,
                    base + (y + slice_h - 1) as u32,
                    base + (x + slice_w - 1) as u32,
                    0,
                ];
                if i + 2 < image.ranges.len() {
                    let following = &image.ranges[i + 2];
                    for at in range.end + 1..following.start - 1 {
                        ensure!(
                            tokens[at] == markers.newline,
                            "unexpected token inside canvas slice row gap"
                        );
                        output[at] = [
                            base,
                            base + (y + slice_h - 1) as u32,
                            base + width as u32,
                            0,
                        ];
                    }
                }
            }
        }
        base = next;
    }
    while output.len() < tokens.len() {
        output.push([base; 4]);
        base += 1;
    }
    Ok(Coordinates {
        positions: output,
        next_text: base,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn image_halos_and_trailing_newline_share_canvas() -> Result<()> {
        let markers = Markers {
            image_start: 10,
            image_end: 11,
            slice_start: 12,
            slice_end: 13,
            newline: 198,
        };
        let plan = ImagePlan {
            thumbnail: [112, 112],
            grid: None,
            refined: None,
        };
        let span = ImageSpan {
            plan,
            ranges: vec![2..6],
        };
        let p = positions(&[100, 10, 20, 20, 20, 20, 11, 198, 200], &[span], markers)?;
        assert_eq!(
            p.positions,
            vec![
                [0; 4],
                [1, 0, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 2, 0],
                [1, 2, 1, 0],
                [1, 2, 2, 0],
                [1, 3, 3, 0],
                [1, 1, 1, 0],
                [4; 4]
            ]
        );
        assert_eq!(p.next_text, 5);
        Ok(())
    }
}
