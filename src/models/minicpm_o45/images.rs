//! MiniCPM-o 14-pixel image slices, each resampled into 64 visual tokens.
use anyhow::{Result, ensure};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImagePlan {
    /// Pixel dimensions, in width/height order.
    pub thumbnail: [usize; 2],
    /// Slice grid, in column/row order; absent for unsliced inputs.
    pub grid: Option<[usize; 2]>,
    pub refined: Option<[usize; 2]>,
}

fn divisible(value: f64, divisor: usize) -> usize {
    ((value / divisor as f64).round_ties_even() as usize).max(1) * divisor
}

fn resize(width: f64, height: f64, upscale: bool) -> [usize; 2] {
    let (width, height) = if width * height > 448.0 * 448.0 || upscale {
        let ratio = width / height;
        let height = (448.0 / ratio.sqrt()).floor();
        ((height * ratio).floor(), height)
    } else {
        (width, height)
    };
    [divisible(width, 14), divisible(height, 14)]
}

impl ImagePlan {
    pub fn new(width: usize, height: usize, max_slices: usize) -> Result<Self> {
        ensure!(
            width > 0 && height > 0 && width <= 16384 && height <= 16384,
            "image dimensions must be within 1..=16384"
        );
        ensure!(
            (1..=36).contains(&max_slices),
            "invalid MiniCPM slice limit"
        );
        let count = (width * height).div_ceil(448 * 448).min(max_slices);
        let grid = if count <= 1 {
            None
        } else {
            let ratio = (width as f64 / height as f64).ln();
            let mut best = [1, 1];
            let mut error = f64::INFINITY;
            for pieces in (count - 1)..=(count + 1) {
                if pieces <= 1 || pieces > max_slices {
                    continue;
                }
                for rows in 1..=pieces {
                    if !pieces.is_multiple_of(rows) {
                        continue;
                    }
                    let columns = pieces / rows;
                    let candidate = (ratio - (columns as f64 / rows as f64).ln()).abs();
                    if candidate < error || (candidate == error && rows > best[1]) {
                        best = [columns, rows];
                        error = candidate;
                    }
                }
            }
            Some(best)
        };
        let thumbnail = resize(width as f64, height as f64, grid.is_none());
        let refined = grid.map(|[columns, rows]| {
            let [w, h] = resize(
                divisible(width as f64, columns) as f64 / columns as f64,
                divisible(height as f64, rows) as f64 / rows as f64,
                true,
            );
            [w * columns, h * rows]
        });
        Ok(Self {
            thumbnail,
            grid,
            refined,
        })
    }

    /// Thumbnail first, followed by row-major slices, all in pixel dimensions.
    pub fn slice_dimensions(&self) -> Vec<[usize; 2]> {
        let mut sizes = vec![self.thumbnail];
        if let (Some([columns, rows]), Some([width, height])) = (self.grid, self.refined) {
            sizes.extend(std::iter::repeat_n(
                [width / columns, height / rows],
                columns * rows,
            ));
        }
        sizes
    }

    pub fn visual_tokens(&self) -> usize {
        self.slice_dimensions().len() * 64
    }

    pub fn placeholder(&self, image_id: usize, use_image_id: bool) -> String {
        let mut text = if use_image_id {
            format!("<image_id>{image_id}</image_id>")
        } else {
            String::new()
        };
        text.push_str("<image>");
        text.push_str(&"<unk>".repeat(64));
        text.push_str("</image>");
        if let Some([columns, rows]) = self.grid {
            for row in 0..rows {
                if row > 0 {
                    text.push('\n');
                }
                for _ in 0..columns {
                    text.push_str("<slice>");
                    text.push_str(&"<unk>".repeat(64));
                    text.push_str("</slice>");
                }
            }
        }
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn adaptive_patches_keep_fixed_resampler_query_counts() -> Result<()> {
        for (width, height) in [(1, 1), (448, 448), (777, 555), (555, 777), (2048, 512)] {
            for max_slices in [1, 9] {
                let plan = ImagePlan::new(width, height, max_slices)?;
                let sizes = plan.slice_dimensions();
                assert!(
                    sizes
                        .iter()
                        .all(|[w, h]| w.is_multiple_of(14) && h.is_multiple_of(14))
                );
                assert_eq!(plan.visual_tokens(), sizes.len() * 64);
                let placeholder = plan.placeholder(2, true);
                assert_eq!(placeholder.matches("<unk>").count(), plan.visual_tokens());
                assert!(placeholder.starts_with("<image_id>2</image_id><image>"));
            }
        }
        assert!(ImagePlan::new(0, 448, 9).is_err());
        Ok(())
    }
}
