//! Duplex acoustic lookahead buffering and checkpoint PCM packet boundaries.
use anyhow::{Context, Result, ensure};
#[derive(Clone, Default)]
pub struct Schedule {
    pending: Vec<u32>,
    finished: bool,
}
pub struct Block {
    pub codes: Vec<u32>,
    pub last: bool,
}
pub struct Audio {
    pub pcm16: Vec<u8>,
    pub speech_samples: usize,
    pub padding_samples: usize,
}
impl Schedule {
    pub fn buffered_codes(&self) -> usize {
        self.pending.len()
    }
    pub fn finished(&self) -> bool {
        self.finished
    }
    pub fn reset(&mut self) {
        self.pending.clear();
        self.finished = false;
    }
    /// A first-unit flush may consume a shorter body, retaining three lookahead
    /// codes. Turn completion flushes that lookahead through a final codec call.
    pub fn push(&mut self, codes: &[u32], last: bool, force_flush: bool) -> Result<Vec<Block>> {
        ensure!(
            !self.finished && codes.iter().all(|&v| v < 6561),
            "invalid codes or finished duplex acoustic schedule"
        );
        self.pending.extend_from_slice(codes);
        let threshold = if force_flush { 8 } else { 28 };
        let mut blocks = Vec::new();
        while self.pending.len() >= threshold {
            let count = self.pending.len().min(28);
            blocks.push(Block {
                codes: self.pending[..count].to_vec(),
                last: false,
            });
            self.pending.drain(..25.min(count - 3));
        }
        if last {
            if !self.pending.is_empty() {
                blocks.push(Block {
                    codes: std::mem::take(&mut self.pending),
                    last: true,
                });
            }
            self.finished = true;
        }
        Ok(blocks)
    }
}
/// Follow checkpoint Token2wav's 32767/truncate conversion, then its unit-level
/// leading-silence rule. Keep padding separate from actual synthesis duration.
pub fn packet(wave: &[f32], last: bool) -> Result<Audio> {
    ensure!(
        wave.iter().all(|v| v.is_finite()),
        "nonfinite duplex waveform"
    );
    let padding_samples = if last || wave.is_empty() {
        0
    } else {
        24000usize.saturating_sub(wave.len())
    };
    let size = wave
        .len()
        .checked_add(padding_samples)
        .and_then(|n| n.checked_mul(2))
        .context("duplex PCM packet extent overflow")?;
    let mut pcm16 = Vec::with_capacity(size);
    pcm16.resize(padding_samples * 2, 0);
    for &value in wave {
        pcm16.extend_from_slice(&((value.clamp(-1., 1.) * 32767.) as i16).to_le_bytes());
    }
    Ok(Audio {
        pcm16,
        speech_samples: wave.len(),
        padding_samples,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn failed_input_preserves_lookahead_and_reset_reopens_schedule() -> Result<()> {
        let mut schedule = Schedule::default();
        assert!(schedule.push(&[1, 2, 3], false, false)?.is_empty());
        assert!(schedule.push(&[6561], false, false).is_err());
        assert_eq!(schedule.buffered_codes(), 3);
        let blocks = schedule.push(&[], true, false)?;
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].codes, [1, 2, 3]);
        assert!(blocks[0].last);
        assert!(schedule.push(&[], false, false).is_err());
        schedule.reset();
        assert!(!schedule.finished());
        assert_eq!(schedule.buffered_codes(), 0);
        Ok(())
    }
    #[test]
    fn padding_and_checkpoint_quantization_are_separate_from_speech() -> Result<()> {
        let audio = packet(&[-1., 0., 1.], false)?;
        assert_eq!((audio.speech_samples, audio.padding_samples), (3, 23997));
        assert_eq!(
            &audio.pcm16[audio.pcm16.len() - 6..],
            &[-32767i16, 0, 32767]
                .into_iter()
                .flat_map(i16::to_le_bytes)
                .collect::<Vec<_>>()
        );
        assert!(packet(&[], false)?.pcm16.is_empty());
        assert_eq!(packet(&[0.; 3], true)?.padding_samples, 0);
        assert!(packet(&[f32::NAN], true).is_err());
        Ok(())
    }
}
