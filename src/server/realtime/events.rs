//! Realtime response envelopes and separate language/code/sample accounting.
use super::*;
use base64::{Engine as _, engine::general_purpose::STANDARD};
pub(super) struct Events {
    sender: tokio::sync::mpsc::Sender<Json>,
    session: String,
    sequence: u64,
    response: Option<String>,
    transcript: String,
    language_tokens: usize,
    speech_codes: usize,
    speech_samples: usize,
    padding_samples: usize,
    ga: bool,
    synthesis_seconds: f64,
}
impl Events {
    pub fn new(sender: tokio::sync::mpsc::Sender<Json>, ga: bool) -> Self {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        Self {
            sender,
            session: format!("sess-lia-{id}"),
            sequence: 0,
            response: None,
            transcript: String::new(),
            language_tokens: 0,
            speech_codes: 0,
            speech_samples: 0,
            padding_samples: 0,
            ga,
            synthesis_seconds: 0.,
        }
    }
    pub fn send(&mut self, mut event: Json) -> Result<()> {
        self.sequence += 1;
        if self.ga {
            let name = match event["type"].as_str() {
                Some("response.audio.delta") => Some("response.output_audio.delta"),
                Some("response.audio.done") => Some("response.output_audio.done"),
                Some("response.audio_transcript.delta") => {
                    Some("response.output_audio_transcript.delta")
                }
                Some("response.audio_transcript.done") => {
                    Some("response.output_audio_transcript.done")
                }
                _ => None,
            };
            if let Some(name) = name {
                event["type"] = name.into();
            }
        }
        event["event_id"] = format!("{}-{}", self.session, self.sequence).into();
        event["server_event_seq"] = self.sequence.into();
        self.sender
            .blocking_send(event)
            .map_err(|_| anyhow!("realtime client disconnected"))
    }
    pub fn error(&mut self, error: &anyhow::Error, source: &Json) -> Result<()> {
        self.send(
            json!({"type":"error","error":{"type":"invalid_request_error",
            "message":format!("{error:#}"),"event_id":source.get("event_id")}}),
        )
    }
    pub fn opened(&mut self, preparation_ms: f64) -> Result<()> {
        let session = json!({"id":self.session,"model":"lia.rs","modalities":["text","audio"],
            "input_audio_format":"pcm16","input_audio_sample_rate":16000,
            "output_audio_format":"pcm16","output_audio_sample_rate":24000,"turn_detection":null});
        self.send(
            json!({"type":"session.created","session":session,"preparation_ms":preparation_ms}),
        )?;
        self.send(json!({"type":"session.updated","session":session}))
    }
    pub fn unit(&mut self, output: &crate::inference::duplex::Output, seconds: f64) -> Result<()> {
        self.send(json!({"type":"lia.duplex.unit","decisions":output.language.decision_tokens,
            "is_listen":output.language.is_listen,"end_of_turn":output.language.end_of_turn,
            "language_position":output.language.end_position,"seconds":seconds,
            "speech_codes":output.speech_codes,"speech_sampled_tokens":output.speech_sampled_tokens,
            "speech_samples":output.audio.speech_samples,"padding_samples":output.audio.padding_samples,
            "timings":{"input_seconds":output.timings.input_seconds,"language_seconds":output.timings.language_seconds,
                "speech_prefill_seconds":output.timings.speech_prefill_seconds,"speech_decode_seconds":output.timings.speech_decode_seconds,
                "acoustic_seconds":output.timings.acoustic_seconds}}))?;
        if self.response.is_none() && (!output.text.is_empty() || !output.audio.pcm16.is_empty()) {
            let id = format!("resp-{}-{}", self.session, self.sequence);
            self.response = Some(id.clone());
            self.send(json!({"type":"response.created","response":{"id":id,"status":"in_progress","output":[]}}))?;
            self.send(json!({"type":"response.output_item.added","response_id":id,"output_index":0,
                "item":{"id":format!("item-{id}"),"type":"message","role":"assistant","status":"in_progress","content":[]}}))?;
            self.send(json!({"type":"response.content_part.added","response_id":id,"item_id":format!("item-{id}"),
                "output_index":0,"content_index":0,"part":{"type":"audio","transcript":""}}))?;
        }
        if let Some(id) = self.response.clone() {
            self.language_tokens += output.language.decision_tokens.len();
            self.speech_codes += output.speech_codes;
            self.speech_samples += output.audio.speech_samples;
            self.padding_samples += output.audio.padding_samples;
            self.synthesis_seconds += output.timings.speech_prefill_seconds
                + output.timings.speech_decode_seconds
                + output.timings.acoustic_seconds;
            if !output.text.is_empty() {
                self.transcript.push_str(&output.text);
                self.send(json!({"type":"response.audio_transcript.delta","response_id":id,
                    "item_id":format!("item-{id}"),"output_index":0,"content_index":0,"delta":output.text}))?;
            }
            if !output.audio.pcm16.is_empty() {
                self.send(json!({"type":"response.audio.delta","response_id":id,"item_id":format!("item-{id}"),
                    "output_index":0,"content_index":0,"delta":STANDARD.encode(&output.audio.pcm16),
                    "sample_rate":24000,"speech_samples":output.audio.speech_samples,"padding_samples":output.audio.padding_samples}))?;
            }
        }
        if output.language.end_of_turn {
            self.done(false)?;
        }
        Ok(())
    }
    pub fn done(&mut self, cancelled: bool) -> Result<()> {
        if let Some(id) = self.response.take() {
            self.send(json!({"type":"response.audio.done","response_id":id,"item_id":format!("item-{id}"),"output_index":0,"content_index":0}))?;
            self.send(json!({"type":"response.audio_transcript.done","response_id":id,"item_id":format!("item-{id}"),
                "output_index":0,"content_index":0,"transcript":self.transcript}))?;
            self.send(json!({"type":"response.content_part.done","response_id":id,"item_id":format!("item-{id}"),
                "output_index":0,"content_index":0,"part":{"type":"audio","transcript":self.transcript}}))?;
            self.send(
                json!({"type":"response.output_item.done","response_id":id,"output_index":0,
                "item":{"id":format!("item-{id}"),"type":"message","role":"assistant",
                    "status":if cancelled {"incomplete"} else {"completed"},
                    "content":[{"type":"audio","transcript":self.transcript}]}}),
            )?;
            self.send(json!({"type":"response.done","response":{"id":id,
                "status":if cancelled {"cancelled"} else {"completed"},
                "output":[{"id":format!("item-{id}"),"type":"message","role":"assistant",
                    "content":[{"type":"audio","transcript":self.transcript}]}],
                "usage":{"output_tokens":self.language_tokens},
                "audio_usage":{"speech_codes":self.speech_codes,"speech_samples":self.speech_samples,"padding_samples":self.padding_samples,
                    "synthesis_seconds":self.synthesis_seconds,
                    "real_time_factor":if self.speech_samples == 0 { None } else { Some(self.synthesis_seconds / (self.speech_samples as f64 / 24000.)) }}}}))?;
        }
        self.transcript.clear();
        self.language_tokens = 0;
        self.speech_codes = 0;
        self.speech_samples = 0;
        self.padding_samples = 0;
        self.synthesis_seconds = 0.;
        Ok(())
    }
}
