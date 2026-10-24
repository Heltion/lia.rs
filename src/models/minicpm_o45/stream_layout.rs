//! Frame selection for the checkpoint encoder's packed two-rate streaming cache.
use anyhow::{Context, Result, ensure};
pub(crate) struct EncoderRetention {
    pub low: Vec<u32>,
    pub high: Vec<u32>,
}
pub(crate) fn retention_indices(total: usize, prefix: usize, suffix: usize) -> Result<Vec<u32>> {
    ensure!(
        total <= super::MAX_MEL_FRAMES && prefix <= total && suffix > 0,
        "invalid acoustic cache retention range"
    );
    let limit = prefix
        .checked_add(suffix)
        .context("acoustic retention length overflow")?;
    ensure!(
        limit <= super::MAX_MEL_FRAMES,
        "acoustic retention exceeds capacity"
    );
    Ok(if total <= limit {
        (0..total).map(|i| i as u32).collect()
    } else {
        (0..prefix)
            .chain(total - suffix..total)
            .map(|i| i as u32)
            .collect()
    })
}
pub(crate) fn encoder_retention(
    code_frames: usize,
    prefix_frames: usize,
    suffix_frames: usize,
) -> Result<EncoderRetention> {
    ensure!(
        code_frames <= super::MAX_CODEC_CODES
            && prefix_frames.is_multiple_of(2)
            && suffix_frames.is_multiple_of(2),
        "encoder retention requires aligned two-rate frames"
    );
    let high = retention_indices(code_frames * 2, prefix_frames, suffix_frames)?;
    // Upstream repeats each low-rate history along time to pack it beside the
    // high-rate layers, trims that packed representation, then takes its first half.
    let low = high[..high.len() / 2]
        .iter()
        .map(|&i| i % code_frames as u32)
        .collect();
    Ok(EncoderRetention { low, high })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_encoder_selection_matches_repeat_trim_and_half_unpack() -> Result<()> {
        let plan = encoder_retention(225, 300, 100)?;
        assert_eq!(plan.low, (0..200).collect::<Vec<_>>());
        assert_eq!(plan.high, (0..300).chain(350..450).collect::<Vec<_>>());
        let plan = encoder_retention(55, 0, 100)?;
        assert_eq!(plan.high, (10..110).collect::<Vec<_>>());
        assert_eq!(plan.low, (10..55).chain(0..5).collect::<Vec<_>>());
        let plan = encoder_retention(175, 300, 100)?;
        assert_eq!(plan.low, (0..175).collect::<Vec<_>>());
        assert_eq!(plan.high, (0..350).collect::<Vec<_>>());
        Ok(())
    }
    #[test]
    fn retention_checks_capacity_and_alignment() -> Result<()> {
        assert!(encoder_retention(10, 3, 4).is_err());
        assert!(encoder_retention(10, 0, 3).is_err());
        assert!(retention_indices(8, 9, 2).is_err());
        assert!(retention_indices(8, 0, 0).is_err());
        assert!(retention_indices(8, 8, usize::MAX).is_err());
        assert!(encoder_retention(0, 0, 100)?.low.is_empty());
        assert_eq!(
            retention_indices(128, 0, 100)?,
            (28..128).collect::<Vec<_>>()
        );
        Ok(())
    }
}
