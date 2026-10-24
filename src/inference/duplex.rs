//! Execute one native duplex unit on the existing causal language cache.
use super::*;
use crate::models::minicpm_o45::{
    duplex::{Control, UNIT, UNIT_END},
    speech_text::Condition,
};
use anyhow::{Context, ensure};
mod conversation;
mod frames;
pub use conversation::{Conversation, Output, Timings};

pub struct Unit {
    pub decision_tokens: Vec<u32>,
    pub condition: Condition,
    pub is_listen: bool,
    pub end_of_turn: bool,
    pub start_position: usize,
    pub end_position: usize,
}
impl Unit {
    /// Apply the native vLLM-Omni speech handoff boundary. A listening unit may
    /// still contain the final speech rows and must not be discarded wholesale.
    pub fn prepare_speech_handoff(&mut self, control: &mut Control) -> Result<()> {
        ensure!(!control.active(), "cannot hand off an active duplex unit");
        if self.condition.finish_duplex_handoff()? {
            self.end_of_turn = true;
            control.finish_turn()?;
        }
        Ok(())
    }
}
impl Engine {
    /// Append <unit>, already encoded input rows, native policy decisions, and
    /// </unit>. A failed execution requires restoring or resetting both states.
    pub fn duplex_unit(
        &mut self,
        control: &mut Control,
        input: &[f32],
        mut sample: impl FnMut(&[f32]) -> Result<u32>,
    ) -> Result<Unit> {
        ensure!(
            !control.active() && self.geometry.qwen3 && !self.geometry.mtp && self.paging.is_none(),
            "invalid duplex language state"
        );
        ensure!(
            !input.is_empty()
                && input.len().is_multiple_of(self.geometry.hidden)
                && input.iter().all(|v| v.is_finite()),
            "invalid duplex input embeddings"
        );
        let rows = input.len() / self.geometry.hidden;
        ensure!(
            rows <= crate::gpu::MAX_PREFILL_BATCH
                && rows + control.limit() + 2 <= self.capacity - self.position,
            "duplex unit exceeds language batch or context capacity"
        );
        let start_position = self.position;
        control.begin()?;
        self.step(UNIT, false)
            .context("feeding duplex unit opener")?;
        let logits = self
            .forward_embeddings(input, control.needs_sample())
            .context("feeding duplex input rows")?;
        let mut logits = if logits.is_empty() {
            Vec::new()
        } else {
            logits[logits.len() - self.vocabulary_size()..].to_vec()
        };
        let mut decision_tokens = Vec::new();
        let mut condition = Condition {
            tokens: Vec::new(),
            hidden: Vec::new(),
        };
        let last = loop {
            let selected = if control.needs_sample() {
                Some(sample(&logits)?)
            } else {
                None
            };
            let decision = control.step(selected)?;
            decision_tokens.push(decision.token);
            if decision.stop {
                self.step(decision.token, false)
                    .context("feeding duplex unit terminator")?;
                break decision;
            }
            logits = self
                .forward(decision.token)
                .context("feeding duplex generated token")?;
            if decision.collect_hidden {
                condition.tokens.push(decision.token);
                condition.hidden.extend(self.current_hidden_state()?);
            }
        };
        self.step(UNIT_END, false)
            .context("feeding duplex unit closure")?;
        Ok(Unit {
            decision_tokens,
            condition,
            is_listen: last.is_listen,
            end_of_turn: last.end_of_turn,
            start_position,
            end_position: self.position,
        })
    }
}
