//! Thumbnail and spatial slice planning for MiniCPM's variable-resolution encoder.
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
    [divisible(width, 56), divisible(height, 56)]
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
        self.slice_dimensions()
            .iter()
            .map(|[w, h]| w / 56 * (h / 56))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complete_patch_groups_and_limits() -> Result<()> {
        for (w, h) in [
            (1, 1),
            (448, 448),
            (1024, 768),
            (768, 1024),
            (1600, 200),
            (16384, 16384),
        ] {
            for maximum in [1, 9, 36] {
                let plan = ImagePlan::new(w, h, maximum)?;
                let sizes = plan.slice_dimensions();
                assert!(!sizes.is_empty() && sizes.len() <= maximum + 1);
                assert!(
                    sizes
                        .iter()
                        .all(|[w, h]| *w > 0 && *h > 0 && w % 56 == 0 && h % 56 == 0)
                );
                assert!(plan.visual_tokens() > 0);
            }
        }
        assert!(ImagePlan::new(0, 448, 9).is_err());
        assert!(ImagePlan::new(448, 448, 0).is_err());
        Ok(())
    }
}
