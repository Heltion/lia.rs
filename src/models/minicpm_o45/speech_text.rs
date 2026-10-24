//! Pair spoken text tokens with their own post-forward language hidden states.
use anyhow::{Result, ensure};
pub struct Condition {
    pub tokens: Vec<u32>,
    pub hidden: Vec<f32>,
}
impl Condition {
    /// Keep the trained turn-stop row, excluding any subsequently decoded tail.
    /// Returns whether this handoff closes the native duplex speech turn.
    pub fn finish_duplex_handoff(&mut self) -> Result<bool> {
        ensure!(
            self.hidden.len()
                == self
                    .tokens
                    .len()
                    .checked_mul(4096)
                    .ok_or_else(|| anyhow::anyhow!("speech condition size overflow"))?,
            "speech tokens and hidden rows are misaligned"
        );
        let Some(index) = self
            .tokens
            .iter()
            .position(|&id| id == super::duplex::TURN_EOS)
        else {
            return Ok(false);
        };
        self.tokens.truncate(index + 1);
        self.hidden.truncate((index + 1) * 4096);
        Ok(true)
    }
}
pub struct Text {
    bos: u32,
    eos: u32,
    open: bool,
    closed: bool,
    condition: Condition,
}
impl Text {
    pub fn new(bos: u32, eos: u32) -> Result<Self> {
        ensure!(bos != eos, "speech text boundary tokens must differ");
        Ok(Self {
            bos,
            eos,
            open: false,
            closed: false,
            condition: Condition {
                tokens: Vec::new(),
                hidden: Vec::new(),
            },
        })
    }
    pub fn needs_hidden(&self, token: u32) -> bool {
        self.open && token != self.bos && token != self.eos
    }
    /// Observe after forwarding this token, rather than pairing it with its predictor's hidden row.
    pub fn observe(&mut self, token: u32, hidden: &[f32]) -> Result<()> {
        if token == self.bos {
            self.condition.tokens.clear();
            self.condition.hidden.clear();
            self.open = true;
            self.closed = false;
        } else if token == self.eos {
            if self.open {
                self.open = false;
                self.closed = true;
            }
        } else if self.open {
            ensure!(
                hidden.len() == 4096 && hidden.iter().all(|v| v.is_finite()),
                "invalid spoken-token hidden row"
            );
            self.condition.tokens.push(token);
            self.condition.hidden.extend_from_slice(hidden);
        }
        Ok(())
    }
    pub fn finish(self) -> Result<Condition> {
        ensure!(
            self.closed && !self.condition.tokens.is_empty(),
            "speech text is empty or lacks its closing boundary"
        );
        Ok(self.condition)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn duplex_handoff_keeps_stop_hidden_and_drops_stale_tail() -> Result<()> {
        let mut condition = Condition {
            tokens: vec![42, super::super::duplex::TURN_EOS, 43],
            hidden: [vec![1.; 4096], vec![2.; 4096], vec![3.; 4096]].concat(),
        };
        assert!(condition.finish_duplex_handoff()?);
        assert_eq!(condition.tokens, [42, super::super::duplex::TURN_EOS]);
        assert_eq!(condition.hidden.len(), 8192);
        assert!(condition.hidden[4096..].iter().all(|&v| v == 2.));
        assert!(condition.finish_duplex_handoff()?);
        let mut empty = Condition {
            tokens: vec![],
            hidden: vec![],
        };
        assert!(!empty.finish_duplex_handoff()?);
        empty.tokens.push(42);
        assert!(empty.finish_duplex_handoff().is_err());
        assert_eq!(empty.tokens, [42]);
        Ok(())
    }
    #[test]
    fn boundaries_exclude_controls_and_thinking_and_keep_token_alignment() -> Result<()> {
        let mut text = Text::new(151703, 151704)?;
        text.observe(77, &[])?;
        text.observe(151703, &[])?;
        assert!(text.needs_hidden(10));
        assert!(!text.needs_hidden(151704));
        text.observe(10, &[1.; 4096])?;
        text.observe(11, &[2.; 4096])?;
        text.observe(151704, &[])?;
        text.observe(151645, &[])?;
        let result = text.finish()?;
        assert_eq!(result.tokens, [10, 11]);
        assert_eq!(&result.hidden[..4096], &[1.; 4096]);
        assert_eq!(&result.hidden[4096..], &[2.; 4096]);
        let mut text = Text::new(1, 2)?;
        text.observe(1, &[])?;
        assert!(text.observe(3, &[0.; 4095]).is_err());
        assert!(text.finish().is_err());
        let mut text = Text::new(1, 2)?;
        text.observe(1, &[])?;
        text.observe(3, &[0.; 4096])?;
        text.observe(2, &[])?;
        text.observe(1, &[])?;
        text.observe(4, &[1.; 4096])?;
        text.observe(2, &[])?;
        assert_eq!(text.finish()?.tokens, [4]);
        Ok(())
    }
}
