//! Explicit native duplex unit boundaries and listen/speak transitions.
use anyhow::{Result, ensure};
pub mod audio;
pub mod context;
pub mod drain;
pub const UNIT: u32 = 151683;
pub const UNIT_END: u32 = 151684;
pub const LISTEN: u32 = 151705;
pub const SPEAK: u32 = 151706;
pub const TTS_BOS: u32 = 151703;
pub const TURN_EOS: u32 = 151717;
pub const CHUNK_EOS: u32 = 151718;
pub const CHUNK_TTS_EOS: u32 = 151721;

#[derive(Clone)]
pub struct Control {
    limit: usize,
    force_listen_count: usize,
    units: usize,
    turn_ended: bool,
    active: bool,
    index: usize,
    forced: bool,
    end_of_turn: bool,
}
#[derive(Debug)]
pub struct Decision {
    pub token: u32,
    pub collect_hidden: bool,
    pub stop: bool,
    pub is_listen: bool,
    pub end_of_turn: bool,
}
impl Control {
    pub fn new(limit: usize, force_listen_count: usize, turn_ended: bool) -> Result<Self> {
        ensure!(
            (1..=4096).contains(&limit),
            "invalid duplex unit token limit"
        );
        Ok(Self {
            limit,
            force_listen_count,
            units: 0,
            turn_ended,
            active: false,
            index: 0,
            forced: false,
            end_of_turn: false,
        })
    }
    /// Start a unit after its input embeddings have been appended to language KV.
    pub fn begin(&mut self) -> Result<()> {
        ensure!(!self.active, "previous duplex unit is still active");
        self.forced = self.units < self.force_listen_count;
        self.units = self
            .units
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("duplex unit counter overflow"))?;
        self.index = 0;
        self.end_of_turn = false;
        self.active = true;
        Ok(())
    }
    pub fn needs_sample(&self) -> bool {
        self.active && self.index < self.limit - 1 && !self.forced
    }
    pub fn limit(&self) -> usize {
        self.limit
    }
    pub fn turn_ended(&self) -> bool {
        self.turn_ended
    }
    pub fn active(&self) -> bool {
        self.active
    }
    /// Latch a completed speech handoff after its language unit has closed.
    pub fn finish_turn(&mut self) -> Result<()> {
        ensure!(!self.active, "cannot finish a turn during a duplex unit");
        self.turn_ended = true;
        Ok(())
    }
    /// Every returned token is fed to language KV. A stopping decision is then
    /// followed by </unit>; only marked rows enter the speech-conditioning stream.
    pub fn step(&mut self, sampled: Option<u32>) -> Result<Decision> {
        ensure!(self.active, "duplex unit is not active");
        ensure!(
            sampled.is_some() == self.needs_sample(),
            "duplex draw supplied at a forced boundary or missing for sampling"
        );
        ensure!(
            sampled.is_none_or(|v| v < 151748),
            "invalid duplex vocabulary token"
        );
        let token = if self.index == self.limit - 1 {
            CHUNK_EOS
        } else if self.forced {
            LISTEN
        } else {
            let token = sampled.expect("sample presence checked");
            if token == LISTEN && !self.turn_ended {
                TTS_BOS
            } else {
                token
            }
        };
        let stop = [LISTEN, CHUNK_EOS, CHUNK_TTS_EOS].contains(&token);
        let collect_hidden = !stop && self.index != 0;
        if !stop {
            self.end_of_turn = token == TURN_EOS;
            self.turn_ended = self.end_of_turn;
        }
        self.index += 1;
        if stop {
            self.active = false;
        }
        Ok(Decision {
            token,
            collect_hidden,
            stop,
            is_listen: token == LISTEN,
            end_of_turn: self.end_of_turn,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_decisions_preserve_state_and_forced_boundaries_do_not_draw() -> Result<()> {
        let mut control = Control::new(2, 0, true)?;
        assert!(control.step(Some(SPEAK)).is_err());
        control.begin()?;
        assert!(control.begin().is_err());
        assert!(control.step(None).is_err());
        assert!(control.step(Some(151748)).is_err());
        assert!(control.needs_sample());
        assert!(!control.step(Some(SPEAK))?.collect_hidden);
        assert!(!control.needs_sample());
        assert!(control.step(Some(10)).is_err());
        let end = control.step(None)?;
        assert!(end.stop && end.token == CHUNK_EOS && !control.active());
        control.begin()?;
        assert_eq!(control.step(Some(LISTEN))?.token, TTS_BOS);
        Ok(())
    }
}
