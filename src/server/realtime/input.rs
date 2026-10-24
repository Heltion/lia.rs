//! Validate wire PCM before it can mutate the session's input buffer.
use super::*;
use base64::{Engine as _, engine::general_purpose::STANDARD};
pub(super) fn pcm(event: &Json) -> Result<Vec<f32>> {
    anyhow::ensure!(
        event
            .get("sample_rate_hz")
            .is_none_or(|v| v.as_u64() == Some(16000)),
        "realtime PCM input must use 16000 Hz"
    );
    anyhow::ensure!(
        event
            .get("format")
            .is_none_or(|v| v.as_str() == Some("pcm16")),
        "realtime input format must be pcm16"
    );
    anyhow::ensure!(
        event.get("audio").is_none_or(Json::is_string),
        "audio must be a base64 string"
    );
    let encoded = event.get("audio").and_then(Json::as_str);
    if encoded.is_none()
        && event["video_frames"]
            .as_array()
            .is_some_and(|v| !v.is_empty())
    {
        return Ok(Vec::new());
    }
    let bytes = STANDARD
        .decode(encoded.context("missing base64 audio")?)
        .context("decoding realtime PCM16")?;
    anyhow::ensure!(
        bytes.len().is_multiple_of(2) && bytes.len() <= 256000,
        "realtime PCM16 must contain complete samples and at most eight seconds"
    );
    Ok(bytes
        .chunks_exact(2)
        .map(|v| i16::from_le_bytes([v[0], v[1]]) as f32 / 32768.)
        .collect())
}
pub(super) fn frames(event: &Json) -> Result<Vec<crate::models::minicpm_o45::Prepared>> {
    let Some(value) = event.get("video_frames") else {
        return Ok(Vec::new());
    };
    let frames = value.as_array().context("video_frames must be an array")?;
    anyhow::ensure!(
        frames.len() <= 8,
        "at most eight frames may share a native unit"
    );
    frames
        .iter()
        .enumerate()
        .map(|(index, frame)| {
            let encoded = frame
                .as_str()
                .filter(|v| !v.is_empty())
                .context("invalid video frame")?;
            let bytes = STANDARD
                .decode(encoded)
                .context("decoding native video frame")?;
            let maximum = if frames.len() > 1 && index == 0 { 2 } else { 1 };
            crate::models::minicpm_o45::prepare(&bytes, maximum)
                .with_context(|| format!("preparing native video frame {index}"))
        })
        .collect()
}
