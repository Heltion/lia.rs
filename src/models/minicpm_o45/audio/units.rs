//! Assemble fragmented 16 kHz transport input without treating hop carry as a unit.
use anyhow::{Result, ensure};
use std::collections::VecDeque;
#[derive(Clone, Default)]
pub struct PcmUnits {
    pending: VecDeque<f32>,
    carry: Vec<f32>,
    started: bool,
}
pub struct Unit {
    pub samples: Vec<f32>,
    pub input_samples: usize,
    pub trailing_padding: usize,
}
impl PcmUnits {
    pub fn buffered_samples(&self) -> usize {
        self.pending.len()
    }
    pub fn append(&mut self, samples: &[f32]) -> Result<()> {
        ensure!(
            samples.iter().all(|v| v.is_finite()),
            "nonfinite transport PCM"
        );
        ensure!(
            samples.len() <= 128_000usize.saturating_sub(self.pending.len()),
            "duplex input buffer exceeds eight seconds"
        );
        self.pending.extend(samples);
        Ok(())
    }
    pub fn clear(&mut self) {
        self.pending.clear();
    }
    /// Only commit pads real residual input. The 80-sample first-hop carry is
    /// retained for the next append and never emitted as a spurious extra unit.
    pub fn next(&mut self, commit: bool) -> Option<Unit> {
        if self.pending.len() < 16000 && (!commit || self.pending.is_empty()) {
            return None;
        }
        let count = self.pending.len().min(16000);
        let mut samples = if self.started {
            std::mem::take(&mut self.carry)
        } else {
            vec![0.; 560]
        };
        samples.extend(self.pending.drain(..count));
        samples.resize(samples.len() + 16000 - count, 0.);
        self.carry = samples.split_off(if self.started { 16000 } else { 16480 });
        self.started = true;
        Some(Unit {
            samples,
            input_samples: count,
            trailing_padding: 16000 - count,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fragmentation_preserves_carry_and_commit_pads_only_real_tail() -> Result<()> {
        let mut input = PcmUnits::default();
        for _ in 0..4 {
            input.append(&[1.; 3200])?;
            assert!(input.next(false).is_none());
        }
        input.append(&[2.; 3200])?;
        let first = input.next(false).unwrap();
        assert_eq!(first.samples.len(), 16480);
        assert_eq!(&first.samples[..560], &[0.; 560]);
        assert_eq!(&first.samples[560..13360], &[1.; 12800]);
        assert!(input.next(true).is_none());
        input.append(&[3.; 17])?;
        let snapshot = input.clone();
        let last = input.next(true).unwrap();
        assert_eq!(&last.samples[..80], &[2.; 80]);
        assert_eq!(&last.samples[80..97], &[3.; 17]);
        assert_eq!((last.input_samples, last.trailing_padding), (17, 15983));
        assert!(last.samples[97..].iter().all(|&v| v == 0.));
        assert!(input.next(true).is_none());
        assert_eq!(snapshot.buffered_samples(), 17);
        input.append(&[4.; 3])?;
        assert!(input.append(&[f32::NAN]).is_err());
        assert_eq!(input.buffered_samples(), 3);
        input.clear();
        assert_eq!(input.buffered_samples(), 0);
        Ok(())
    }
}
