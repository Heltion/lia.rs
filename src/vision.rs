//! Image decoding and preprocessing for the Qwen vision encoder.
use crate::gguf::{Model, Value};
use anyhow::{Context, Result, anyhow, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use std::io::Cursor;

mod plan;
mod positions;
pub use plan::PositionPlan;
mod huggingface;
mod resize;
pub use positions::ImageGrid;

pub struct PreparedImage {
    pub width: usize,
    pub height: usize,
    /// Channel-major normalized RGB, ready for patch extraction.
    pub pixels: Vec<f32>,
}

pub struct Processor {
    mean: [f32; 3],
    std: [f32; 3],
    min_pixels: usize,
    max_pixels: usize,
}

impl Processor {
    pub fn from_model(model: &Model, min_tokens: usize, max_tokens: usize) -> Result<Self> {
        ensure!(
            matches!(model.metadata.get("clip.projector_type"), Some(Value::String(s)) if s == "qwen3vl_merger"),
            "unsupported vision projector"
        );
        for (key, expected) in [("patch_size", 16), ("spatial_merge_size", 2)] {
            ensure!(
                matches!(model.metadata.get(&format!("clip.vision.{key}")), Some(Value::Unsigned(n)) if *n == expected),
                "unsupported vision {key}"
            );
        }
        ensure!(
            min_tokens > 0 && min_tokens <= max_tokens && max_tokens <= 16384,
            "image token limits must satisfy 1 <= min <= max <= 16384"
        );
        let array = |key: &str| -> Result<[f32; 3]> {
            let Some(Value::Array { values, .. }) = model.metadata.get(key) else {
                return Err(anyhow!("missing vision metadata {key}"));
            };
            ensure!(values.len() == 3, "invalid vision metadata {key}");
            let mut out = [0.0; 3];
            for (dst, src) in out.iter_mut().zip(values) {
                let Value::Float(v) = src else {
                    return Err(anyhow!("invalid {key}"));
                };
                *dst = *v as f32;
                ensure!(dst.is_finite(), "nonfinite vision metadata {key}");
            }
            Ok(out)
        };
        let mean = array("clip.vision.image_mean")?;
        let std = array("clip.vision.image_std")?;
        ensure!(
            std.iter().all(|v| *v > 0.0),
            "vision standard deviations must be positive"
        );
        Ok(Self {
            mean,
            std,
            min_pixels: min_tokens * 1024,
            max_pixels: max_tokens * 1024,
        })
    }

    pub fn dimensions(&self, width: usize, height: usize) -> Result<(usize, usize)> {
        ensure!(
            width > 0 && height > 0 && width <= 16384 && height <= 16384,
            "image dimensions must be between 1 and 16384"
        );
        let align = |x: f32, mode: u8| -> usize {
            let x = x / 32.0;
            (match mode {
                0 => x.round(),
                1 => x.floor(),
                _ => x.ceil(),
            } as usize)
                .max(1)
                * 32
        };
        let (mut w, mut h) = (align(width as f32, 0), align(height as f32, 0));
        let area = (width * height) as f32;
        if w * h > self.max_pixels {
            let scale = (area / self.max_pixels as f32).sqrt();
            w = align(width as f32 / scale, 1);
            h = align(height as f32 / scale, 1);
        } else if w * h < self.min_pixels {
            let scale = (self.min_pixels as f32 / area).sqrt();
            w = align(width as f32 * scale, 2);
            h = align(height as f32 * scale, 2);
        }
        ensure!(
            w.checked_mul(h).is_some_and(|n| n <= 16 * 1024 * 1024),
            "resized image exceeds the pixel limit"
        );
        Ok((w, h))
    }

    pub fn prepare_data_url(&self, url: &str) -> Result<PreparedImage> {
        let (header, data) = url
            .split_once(',')
            .ok_or_else(|| anyhow!("invalid image data URL"))?;
        let format = match header {
            "data:image/png;base64" => image::ImageFormat::Png,
            "data:image/jpeg;base64" => image::ImageFormat::Jpeg,
            _ => return Err(anyhow!("expected a PNG or JPEG base64 data URL")),
        };
        let bytes = STANDARD.decode(data).context("decoding image base64")?;
        ensure!(
            image::guess_format(&bytes)? == format,
            "image MIME type does not match its data"
        );
        self.prepare(&bytes)
    }

    pub fn prepare(&self, bytes: &[u8]) -> Result<PreparedImage> {
        let format = image::guess_format(bytes).context("identifying image format")?;
        ensure!(
            matches!(format, image::ImageFormat::Png | image::ImageFormat::Jpeg),
            "only PNG and JPEG images are supported"
        );
        let mut reader = image::ImageReader::with_format(Cursor::new(bytes), format);
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(16384);
        limits.max_image_height = Some(16384);
        limits.max_alloc = Some(256 * 1024 * 1024);
        reader.limits(limits);
        let decoded = reader
            .decode()
            .context("decoding image pixels")?
            .into_rgb8();
        let (w, h) = (decoded.width() as usize, decoded.height() as usize);
        let (width, height) = self.dimensions(w, h)?;
        let rgb = resize::bicubic(decoded.as_raw(), w, h, width, height);
        let count = width * height;
        let mut pixels = vec![0.0; count * 3];
        for channel in 0..3 {
            for i in 0..count {
                pixels[channel * count + i] =
                    (rgb[i * 3 + channel] as f32 / 255.0 - self.mean[channel]) / self.std[channel];
            }
        }
        Ok(PreparedImage {
            width,
            height,
            pixels,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageEncoder;

    fn processor() -> Processor {
        Processor {
            mean: [0.5, 0.25, 0.5],
            std: [0.5, 0.25, 0.5],
            min_pixels: 256 * 1024,
            max_pixels: 1024 * 1024,
        }
    }

    #[test]
    fn sizing_preserves_reference_rounding_and_patch_groups() -> Result<()> {
        let p = processor();
        for (source, target) in [
            ((733, 237), (928, 320)),
            ((546, 858), (544, 864)),
            ((430, 173), (832, 352)),
            ((4096, 4096), (1024, 1024)),
        ] {
            assert_eq!(p.dimensions(source.0, source.1)?, target);
        }
        assert!(p.dimensions(0, 1).is_err());
        assert!(p.dimensions(16385, 1).is_err());
        Ok(())
    }

    #[test]
    fn data_urls_decode_normalize_and_reject_invalid_data() -> Result<()> {
        let p = processor();
        let mut bytes = Vec::new();
        image::codecs::png::PngEncoder::new(&mut bytes).write_image(
            &[128, 64, 255],
            1,
            1,
            image::ExtendedColorType::Rgb8,
        )?;
        let result = p.prepare_data_url(&format!(
            "data:image/png;base64,{}",
            STANDARD.encode(&bytes)
        ))?;
        let count = result.width * result.height;
        assert_eq!((result.width, result.height), (512, 512));
        for channel in 0..3 {
            let expected =
                ([128, 64, 255][channel] as f32 / 255.0 - p.mean[channel]) / p.std[channel];
            assert!(
                result.pixels[channel * count..(channel + 1) * count]
                    .iter()
                    .all(|v| *v == expected)
            );
        }
        assert!(p.prepare_data_url("https://example.com/image.png").is_err());
        assert!(p.prepare_data_url("data:image/png;base64,!").is_err());
        assert!(
            p.prepare_data_url(&format!(
                "data:image/jpeg;base64,{}",
                STANDARD.encode(bytes)
            ))
            .is_err()
        );
        assert!(p.prepare(b"invalid image").is_err());
        Ok(())
    }
}
