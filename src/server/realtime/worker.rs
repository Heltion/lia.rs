//! One native duplex session owns the serial engine until socket disconnect.
use super::*;
use crate::{inference::duplex::Conversation, models::minicpm_o45::audio::units::PcmUnits};
struct State {
    conversation: Option<Conversation>,
    pending: PcmUnits,
    has_input: bool,
    frames: std::collections::VecDeque<Vec<crate::models::minicpm_o45::Prepared>>,
    events: events::Events,
}
pub(in crate::server) fn run(
    session: &mut Session,
    tokenizer: &Tokenizer,
    path: &str,
    mut channels: Channels,
) -> Result<()> {
    let mut state = State {
        conversation: None,
        pending: PcmUnits::default(),
        has_input: false,
        frames: Default::default(),
        events: events::Events::new(channels.output.clone(), channels.ga),
    };
    session.tokens.clear();
    session.media_sources.clear();
    let result = (|| -> Result<()> {
        anyhow::ensure!(
            session.speech.is_some(),
            "native duplex requires MiniCPM-o 4.5"
        );
        while let Some(command) = channels.input.blocking_recv() {
            let event = command.event;
            if channels.disconnected.load(Ordering::Acquire) {
                break;
            }
            if event["type"] == "session.close" {
                break;
            }
            if let Err(error) =
                state.command(session, tokenizer, path, &channels, &event, command.epoch)
            {
                state.events.error(&error, &event)?;
                if state.conversation.as_ref().is_some_and(|c| !c.valid()) {
                    break;
                }
            }
        }
        Ok(())
    })();
    if let Err(error) = &result {
        let _ = state.events.error(error, &Json::Null);
    }
    if let Some(conversation) = &mut state.conversation {
        conversation.cancel();
    }
    drop(state);
    session.engine.trim_scratch();
    session
        .engine
        .reset()
        .and_then(|()| session.engine.refresh_snapshot(&mut session.checkpoint))
        .context("resetting engine after realtime session")?;
    result
}
impl State {
    fn command(
        &mut self,
        session: &mut Session,
        tokenizer: &Tokenizer,
        path: &str,
        channels: &Channels,
        event: &Json,
        epoch: u64,
    ) -> Result<()> {
        if event["type"] == "session.update" {
            anyhow::ensure!(
                self.conversation.is_none(),
                "updating an active native session is not implemented; open a new session"
            );
            let started = Instant::now();
            self.conversation = Some(config::open(session, tokenizer, path, &event["session"])?);
            return self.events.opened(started.elapsed().as_secs_f64() * 1000.);
        }
        anyhow::ensure!(
            self.conversation.is_some(),
            "send session.update before realtime input"
        );
        match event["type"]
            .as_str()
            .context("missing realtime event type")?
        {
            "input_audio_buffer.append" => {
                let pcm = input::pcm(event)?;
                let frames = input::frames(event)?;
                anyhow::ensure!(
                    frames.is_empty() || self.frames.len() < 8,
                    "native video queue exceeds eight groups"
                );
                if pcm.is_empty() && !frames.is_empty() {
                    anyhow::ensure!(
                        self.pending.buffered_samples() == 0 && self.frames.is_empty(),
                        "complete or clear pending audio before a video-only unit"
                    );
                    let started = Instant::now();
                    let output = self
                        .conversation
                        .as_mut()
                        .context("missing native conversation")?
                        .audio_video_unit(&mut session.engine, tokenizer, &[], &frames, || {
                            channels.disconnected.load(Ordering::Acquire)
                        })?;
                    if channels.epoch.load(Ordering::Acquire) == epoch {
                        self.events.unit(&output, started.elapsed().as_secs_f64())?;
                    }
                    self.has_input = true;
                    return Ok(());
                }
                self.pending.append(&pcm)?;
                if !frames.is_empty() {
                    self.frames.push_back(frames);
                }
                self.has_input |= !pcm.is_empty();
                self.process(session, tokenizer, channels, false, epoch)?;
            }
            "input_audio_buffer.commit" | "input.commit" => {
                anyhow::ensure!(self.has_input, "cannot commit an empty input buffer");
                self.process(session, tokenizer, channels, true, epoch)?;
                self.has_input = false;
                self.events.send(json!({"type":"input_audio_buffer.committed","item_id":format!("input-{}", session.engine.position())}))?;
            }
            "input_audio_buffer.clear" | "input.cancel" => {
                self.pending.clear();
                self.frames.clear();
                self.has_input = false;
                self.events
                    .send(json!({"type":"input_audio_buffer.cleared"}))?;
            }
            "response.create" => {
                if self.pending.buffered_samples() == 0 {
                    self.pending.append(&[0.; 16000])?;
                }
                self.process(session, tokenizer, channels, true, epoch)?;
            }
            "response.cancel" | "barge_in" | "output_audio_buffer.clear" => {
                self.conversation
                    .as_mut()
                    .context("missing native conversation")?
                    .interrupt()?;
                self.events.done(true)?;
                self.events
                    .send(json!({"type":"output_audio_buffer.cleared"}))?;
            }
            "session.heartbeat" => {
                self.events.send(json!({"type":"session.heartbeat"}))?;
            }
            other => anyhow::bail!("unsupported native realtime event: {other}"),
        }
        Ok(())
    }
    fn process(
        &mut self,
        session: &mut Session,
        tokenizer: &Tokenizer,
        channels: &Channels,
        commit: bool,
        epoch: u64,
    ) -> Result<()> {
        while channels.epoch.load(Ordering::Acquire) == epoch {
            let Some(unit) = self.pending.next(commit) else {
                break;
            };
            let started = Instant::now();
            let frames = self.frames.pop_front().unwrap_or_default();
            let output = self
                .conversation
                .as_mut()
                .context("missing native conversation")?
                .audio_video_unit(
                    &mut session.engine,
                    tokenizer,
                    &unit.samples,
                    &frames,
                    || channels.disconnected.load(Ordering::Acquire),
                )?;
            if channels.epoch.load(Ordering::Acquire) == epoch {
                self.events.unit(&output, started.elapsed().as_secs_f64())?;
            }
        }
        Ok(())
    }
}
