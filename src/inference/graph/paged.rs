//! Paged graphs read live tokens, positions and page tables on every replay.
use super::*;

pub(in crate::inference) struct Replay {
    executable: Executable,
    logits: Option<Buffer>,
}

impl Replay {
    pub(in crate::inference) fn new(
        engine: &Engine,
        capture: Capture,
        logits: Option<Buffer>,
    ) -> Result<Self> {
        let graph = capture.end().context("capturing paged execution")?;
        let executable = Executable::instantiate(&graph, engine._context.clone())?;
        engine.graph_stats.borrow_mut()[2] += 1;
        Ok(Self { executable, logits })
    }

    pub(in crate::inference) fn launch(&self, engine: &Engine) -> Result<()> {
        self.executable.launch(&engine.stream)?;
        engine.graph_stats.borrow_mut()[0] += 1;
        Ok(())
    }
    pub(in crate::inference) fn logits(&self) -> Option<&DeviceBuffer<f32>> {
        self.logits.as_deref()
    }
}
