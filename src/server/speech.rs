//! Native spoken-response lifecycle, independent of the language KV cache.
use super::*;
use crate::models::minicpm_o45::{
    PreparedVoice,
    speech_text::{Condition, Text},
};
mod request;
mod response;
mod streaming;
use request::Options;
pub(super) use request::{prepare, requested};
pub(super) struct State {
    directory: String,
    reference: Option<(String, PreparedVoice)>,
}
pub(super) struct Capture {
    options: Options,
    collector: Option<Text>,
    segments: Vec<Condition>,
    setup_ms: f64,
    started: Instant,
}
impl State {
    pub(super) fn new(directory: String) -> Self {
        Self {
            directory,
            reference: None,
        }
    }
    pub(super) fn begin(&mut self, engine: &mut Engine, input: &Json) -> Result<Capture> {
        let options = Options::parse(input)?;
        let started = Instant::now();
        engine.trim_scratch();
        engine.load_speech(&self.directory)?;
        engine.load_codec_encoder(&self.directory)?;
        engine.load_codec_flow(&self.directory)?;
        engine.load_codec_hift(&self.directory)?;
        engine.load_reference_voice_weights(&self.directory)?;
        let bytes = if let Some(bytes) = &options.reference {
            bytes.clone()
        } else {
            std::fs::read(std::path::Path::new(&self.directory).join("assets/HT_ref_audio.wav"))
                .context("reading default reference audio")?
        };
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        hash.update(&bytes);
        hash.update(options.seconds.to_le_bytes());
        let key = format!("{:x}", hash.finalize());
        if self.reference.as_ref().is_none_or(|(old, _)| *old != key) {
            self.reference = Some((key, engine.prepare_voice_wav(&bytes, options.seconds)?));
        }
        engine.trim_scratch();
        Ok(Capture {
            options,
            collector: None,
            segments: Vec::new(),
            setup_ms: started.elapsed().as_secs_f64() * 1000.,
            started,
        })
    }
}
impl Capture {
    /// A selected turn boundary closes conditioning without a language forward:
    /// it is neither a spoken token nor a row requiring its own hidden state.
    pub(super) fn observe_stop(&mut self, token: u32) -> Result<()> {
        if token == 151704 {
            if let Some(mut collector) = self.collector.take() {
                collector.observe(token, &[])?;
                self.segments.push(collector.finish()?);
            }
        }
        Ok(())
    }

    pub(super) fn text(&self) -> bool {
        self.options.text
    }
    pub(super) fn start_prompt(&mut self, prompt: &[u32]) -> Result<()> {
        if prompt.last() == Some(&151703) {
            let mut collector = Text::new(151703, 151704)?;
            collector.observe(151703, &[])?;
            self.collector = Some(collector);
        }
        Ok(())
    }
    pub(super) fn observe(&mut self, engine: &Engine, confirmed: &[u32]) -> Result<()> {
        anyhow::ensure!(
            !engine.mtp_enabled() && confirmed.len() == 1,
            "speech hidden capture requires one confirmed native token"
        );
        let token = confirmed[0];
        if token == 151703 && self.collector.is_none() {
            self.collector = Some(Text::new(151703, 151704)?);
        }
        if let Some(collector) = &mut self.collector {
            let hidden = if collector.needs_hidden(token) {
                engine.current_hidden_state()?
            } else {
                Vec::new()
            };
            collector.observe(token, &hidden)?;
            if token == 151704 {
                self.segments.push(
                    self.collector
                        .take()
                        .context("missing speech collector")?
                        .finish()?,
                );
            }
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selected_tts_stop_closes_existing_spoken_conditioning_without_hidden() -> Result<()> {
        let mut capture = Capture {
            options: Options::parse(&json!({"modalities":["text","audio"]}))?,
            collector: None,
            segments: Vec::new(),
            setup_ms: 0.,
            started: Instant::now(),
        };
        capture.start_prompt(&[151703])?;
        capture
            .collector
            .as_mut()
            .unwrap()
            .observe(10, &[0.25; 4096])?;
        capture.observe_stop(151645)?;
        assert!(capture.collector.is_some());
        assert!(capture.segments.is_empty());
        capture.observe_stop(151704)?;
        assert!(capture.collector.is_none());
        assert_eq!(capture.segments[0].tokens, [10]);
        assert_eq!(capture.segments[0].hidden, [0.25; 4096]);
        capture.observe_stop(151704)?;
        assert_eq!(capture.segments.len(), 1);
        Ok(())
    }
    #[test]
    fn prompt_tts_boundary_opens_collector() -> Result<()> {
        let options = Options::parse(&json!({"modalities":["text","audio"]}))?;
        let mut capture = Capture {
            options,
            collector: None,
            segments: Vec::new(),
            setup_ms: 0.,
            started: Instant::now(),
        };
        capture.start_prompt(&[1, 151703])?;
        assert!(
            capture
                .collector
                .as_ref()
                .is_some_and(|v| v.needs_hidden(10))
        );
        Ok(())
    }
}
