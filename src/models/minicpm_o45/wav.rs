//! Checked little-endian RIFF PCM/IEEE-float decoding without external programs.
use anyhow::{Context, Result, ensure};
mod encode;
pub use encode::pcm16;

pub struct Audio {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}

fn u16_at(bytes: &[u8], at: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(
        bytes
            .get(at..at + 2)
            .context("truncated WAV integer")?
            .try_into()?,
    ))
}
fn u32_at(bytes: &[u8], at: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        bytes
            .get(at..at + 4)
            .context("truncated WAV integer")?
            .try_into()?,
    ))
}

pub fn decode(bytes: &[u8]) -> Result<Audio> {
    ensure!(
        bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "expected little-endian RIFF/WAVE audio"
    );
    let end = usize::try_from(u32_at(bytes, 4)?)?
        .checked_add(8)
        .context("WAV size overflow")?;
    ensure!(end >= 12 && end <= bytes.len(), "truncated WAV container");
    let mut format = None;
    let mut data = None;
    let mut at = 12;
    while at < end {
        ensure!(end - at >= 8, "truncated WAV chunk header");
        let size = usize::try_from(u32_at(bytes, at + 4)?)?;
        let start = at + 8;
        let stop = start.checked_add(size).context("WAV chunk size overflow")?;
        ensure!(stop <= end, "truncated WAV chunk");
        match &bytes[at..at + 4] {
            b"fmt " => {
                ensure!(format.is_none(), "duplicate WAV format");
                format = Some(&bytes[start..stop]);
            }
            b"data" => {
                ensure!(data.is_none(), "duplicate WAV data");
                data = Some(&bytes[start..stop]);
            }
            _ => {}
        }
        at = stop
            .checked_add(size % 2)
            .context("WAV alignment overflow")?;
        ensure!(at <= end, "missing WAV chunk padding");
    }
    let format = format.context("missing WAV format")?;
    ensure!(format.len() >= 16, "truncated WAV format");
    let encoding = u16_at(format, 0)?;
    let channels = usize::from(u16_at(format, 2)?);
    let sample_rate = u32_at(format, 4)?;
    let align = usize::from(u16_at(format, 12)?);
    let bits = u16_at(format, 14)?;
    ensure!(
        (1..=32).contains(&channels) && (8000..=192000).contains(&sample_rate),
        "invalid WAV channels or sample rate"
    );
    ensure!(
        (encoding == 1 && [8, 16, 24, 32].contains(&bits))
            || (encoding == 3 && [32, 64].contains(&bits)),
        "unsupported WAV encoding or bit depth"
    );
    let width = usize::from(bits / 8);
    ensure!(
        align == channels * width && u32_at(format, 8)? == sample_rate * align as u32,
        "inconsistent WAV frame layout"
    );
    let data = data.context("missing WAV sample data")?;
    ensure!(
        !data.is_empty() && data.len().is_multiple_of(align),
        "incomplete WAV sample frames"
    );
    ensure!(
        data.len() / align <= sample_rate as usize * 300,
        "WAV exceeds 300-second input limit"
    );
    let mut samples = Vec::with_capacity(data.len() / align);
    for frame in data.chunks_exact(align) {
        let mut sum = 0.0f64;
        for sample in frame.chunks_exact(width) {
            let value = match (encoding, bits) {
                (1, 8) => (f64::from(sample[0]) - 128.0) / 128.0,
                (1, 16) => f64::from(i16::from_le_bytes(sample.try_into()?)) / 32768.0,
                (1, 24) => {
                    let signed = i32::from_le_bytes([0, sample[0], sample[1], sample[2]]) >> 8;
                    f64::from(signed) / 8388608.0
                }
                (1, 32) => f64::from(i32::from_le_bytes(sample.try_into()?)) / 2147483648.0,
                (3, 32) => f64::from(f32::from_le_bytes(sample.try_into()?)),
                (3, 64) => f64::from_le_bytes(sample.try_into()?),
                _ => unreachable!(),
            };
            ensure!(value.is_finite(), "non-finite WAV sample");
            sum += value;
        }
        let mono = (sum / channels as f64) as f32;
        ensure!(mono.is_finite(), "WAV sample exceeds FP32 range");
        samples.push(mono);
    }
    Ok(Audio {
        sample_rate,
        samples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn wav(bits: u16, channels: u16, data: &[u8]) -> Vec<u8> {
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&(36u32 + data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&16000u32.to_le_bytes());
        let align = channels * bits / 8;
        bytes.extend_from_slice(&(16000u32 * u32::from(align)).to_le_bytes());
        bytes.extend_from_slice(&align.to_le_bytes());
        bytes.extend_from_slice(&bits.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(data);
        bytes
    }
    #[test]
    fn pcm_extremes_stereo_and_truncation() -> Result<()> {
        let bytes = wav(24, 2, &[0, 0, 128, 255, 255, 127, 0, 0, 64, 0, 0, 64]);
        let audio = decode(&bytes)?;
        assert_eq!(audio.sample_rate, 16000);
        assert_eq!(audio.samples, vec![-1.0 / 16777216.0, 0.5]);
        assert!(decode(&bytes[..bytes.len() - 1]).is_err());
        let bytes = wav(16, 1, &[0, 128, 255, 127, 0, 0]);
        assert_eq!(decode(&bytes)?.samples, vec![-1.0, 32767.0 / 32768.0, 0.0]);
        assert!(decode(&wav(16, 1, &[0])).is_err());
        assert!(decode(b"not a WAV").is_err());
        Ok(())
    }
}
