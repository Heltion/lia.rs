//! Patch order and language-model rotary coordinates for merged image grids.
use anyhow::{Result, ensure};

#[derive(Clone, Copy, Debug)]
pub struct ImageGrid {
    width: usize,
    height: usize,
}

impl ImageGrid {
    pub fn new(pixel_width: usize, pixel_height: usize) -> Result<Self> {
        ensure!(
            pixel_width > 0
                && pixel_height > 0
                && pixel_width.is_multiple_of(32)
                && pixel_height.is_multiple_of(32),
            "image grid dimensions must be positive multiples of 32"
        );
        ensure!(
            pixel_width
                .checked_mul(pixel_height)
                .is_some_and(|n| n <= 16 * 1024 * 1024),
            "image grid exceeds pixel limit"
        );
        Ok(Self {
            width: pixel_width / 32,
            height: pixel_height / 32,
        })
    }

    pub fn token_count(self) -> usize {
        self.width * self.height
    }

    /// Encoder patches are grouped in consecutive 2x2 blocks before merging.
    pub fn patch_coordinates(self, patch: usize) -> Result<(usize, usize)> {
        ensure!(
            patch < self.token_count() * 4,
            "patch index outside image grid"
        );
        let group = patch / 4;
        Ok((
            (group % self.width) * 2 + patch % 2,
            (group / self.width) * 2 + (patch % 4) / 2,
        ))
    }

    /// Decoder M-RoPE coordinates in temporal, row, column, unused order.
    /// Cache offsets remain sequential and must not be replaced by these values.
    pub fn decoder_position(self, start: u32, token: usize) -> Result<[u32; 4]> {
        ensure!(token < self.token_count(), "token index outside image grid");
        self.next_position(start)?;
        Ok([
            start,
            start + (token / self.width) as u32,
            start + (token % self.width) as u32,
            0,
        ])
    }

    pub fn next_position(self, start: u32) -> Result<u32> {
        start
            .checked_add(self.width.max(self.height) as u32)
            .ok_or_else(|| anyhow::anyhow!("image rotary position overflow"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merged_patch_order_and_decoder_axes_are_distinct() -> Result<()> {
        let grid = ImageGrid::new(96, 64)?;
        assert_eq!(grid.token_count(), 6);
        let expected = [
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (2, 0),
            (3, 0),
            (2, 1),
            (3, 1),
            (4, 0),
            (5, 0),
            (4, 1),
            (5, 1),
            (0, 2),
            (1, 2),
            (0, 3),
            (1, 3),
        ];
        for (i, xy) in expected.into_iter().enumerate() {
            assert_eq!(grid.patch_coordinates(i)?, xy);
        }
        assert_eq!(grid.decoder_position(11, 0)?, [11, 11, 11, 0]);
        assert_eq!(grid.decoder_position(11, 4)?, [11, 12, 12, 0]);
        assert_eq!(grid.decoder_position(11, 5)?, [11, 12, 13, 0]);
        assert_eq!(grid.next_position(11)?, 14);
        assert!(grid.patch_coordinates(24).is_err());
        assert!(grid.decoder_position(11, 6).is_err());
        Ok(())
    }

    #[test]
    fn boundaries_and_non_square_grids() -> Result<()> {
        assert!(ImageGrid::new(0, 32).is_err());
        assert!(ImageGrid::new(33, 32).is_err());
        assert!(ImageGrid::new(usize::MAX - 31, 32).is_err());
        let grid = ImageGrid::new(928, 320)?;
        assert_eq!(grid.token_count(), 290);
        assert_eq!(grid.next_position(46)?, 75);
        assert_eq!(grid.decoder_position(46, 289)?, [46, 55, 74, 0]);
        assert!(grid.next_position(u32::MAX).is_err());
        let portrait = ImageGrid::new(32, 96)?;
        assert_eq!(portrait.next_position(5)?, 8);
        assert_eq!(portrait.decoder_position(5, 2)?, [5, 7, 5, 0]);
        Ok(())
    }
}
