//! Logical rotary positions are independent of sequential KV cache indices.
use super::ImageGrid;
use anyhow::{Result, ensure};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PositionPlan {
    positions: Vec<[u32; 4]>,
    next: u32,
}

impl PositionPlan {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    pub fn next_text_position(&self) -> u32 {
        self.next
    }

    pub fn append_text(&mut self, count: usize) -> Result<()> {
        ensure!(
            count <= 32768 - self.len(),
            "position plan exceeds context limit"
        );
        let next = self
            .next
            .checked_add(count as u32)
            .ok_or_else(|| anyhow::anyhow!("text rotary position overflow"))?;
        self.positions.extend((self.next..next).map(|p| [p; 4]));
        self.next = next;
        Ok(())
    }

    /// Image start/end markers are ordinary text, appended by the caller.
    pub fn append_image(&mut self, grid: ImageGrid) -> Result<usize> {
        ensure!(
            grid.token_count() <= 32768 - self.len(),
            "position plan exceeds context limit"
        );
        let start = self.len();
        let next = grid.next_position(self.next)?;
        for token in 0..grid.token_count() {
            self.positions
                .push(grid.decoder_position(self.next, token)?);
        }
        self.next = next;
        Ok(start)
    }

    /// Draft and target positions after the prompt advance identically.
    pub fn coordinates(&self, cache_index: usize) -> Result<[u32; 4]> {
        if let Some(position) = self.positions.get(cache_index) {
            return Ok(*position);
        }
        let delta = u32::try_from(cache_index - self.len())?;
        let p = self
            .next
            .checked_add(delta)
            .ok_or_else(|| anyhow::anyhow!("generated rotary position overflow"))?;
        Ok([p; 4])
    }

    pub(crate) fn flattened(&self) -> Vec<u32> {
        self.positions.iter().flatten().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_images_and_drafts_use_continuous_logical_positions() -> Result<()> {
        let mut plan = PositionPlan::new();
        plan.append_text(46)?;
        assert_eq!(plan.append_image(ImageGrid::new(928, 320)?)?, 46);
        plan.append_text(107)?;
        assert_eq!(plan.len(), 443);
        assert_eq!(plan.coordinates(46)?, [46, 46, 46, 0]);
        assert_eq!(plan.coordinates(335)?, [46, 55, 74, 0]);
        assert_eq!(plan.coordinates(336)?, [75; 4]);
        assert_eq!(plan.coordinates(442)?, [181; 4]);
        assert_eq!(plan.coordinates(443)?, [182; 4]);
        assert_eq!(plan.coordinates(446)?, [185; 4]);
        // A second image starts at the logical cursor, not at the KV length.
        assert_eq!(plan.append_image(ImageGrid::new(64, 96)?)?, 443);
        assert_eq!(plan.coordinates(448)?, [182, 184, 183, 0]);
        assert_eq!(plan.coordinates(449)?, [185; 4]);
        Ok(())
    }

    #[test]
    fn text_only_and_limits() -> Result<()> {
        let mut plan = PositionPlan::new();
        assert_eq!(plan.coordinates(0)?, [0; 4]);
        plan.append_text(32768)?;
        assert!(plan.append_text(1).is_err());
        assert!(plan.append_image(ImageGrid::new(32, 32)?).is_err());
        assert_eq!(plan.coordinates(32767)?, [32767; 4]);
        assert_eq!(plan.flattened().len(), 32768 * 4);
        assert!(plan.coordinates(usize::MAX).is_err());
        Ok(())
    }
}
