//! Read static SVG and raster images without a browser or desktop capture.
use anyhow::{Context, Result};
use base64::{Engine, engine::general_purpose::STANDARD};
use image::{ImageFormat, RgbaImage};
use serde_json::{Value, json};
use std::{
    io::{Cursor, Read},
    path::Path,
};
mod animation;
mod protocol;
mod svg;
pub use protocol::serve;

const MAX_FILE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_SIDE: u32 = 2048;
const MAX_PIXELS: u64 = 16 * 1024 * 1024;

pub struct Picture {
    pub png: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub source: String,
    pub svg: bool,
    pub time_seconds: Option<f64>,
}

pub fn read(path: impl AsRef<Path>) -> Result<Picture> {
    read_at(path, None)
}

pub fn read_at(path: impl AsRef<Path>, time: Option<f64>) -> Result<Picture> {
    let path = path
        .as_ref()
        .canonicalize()
        .context("resolving image path")?;
    let file = std::fs::File::open(&path).context("opening image file")?;
    anyhow::ensure!(
        file.metadata()?.is_file(),
        "image path must be a regular file"
    );
    let mut data = Vec::new();
    file.take(MAX_FILE_BYTES + 1)
        .read_to_end(&mut data)
        .context("reading image file")?;
    anyhow::ensure!(
        data.len() as u64 <= MAX_FILE_BYTES,
        "image exceeds 16 MiB input limit"
    );
    let is_svg = path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("svg"));
    let time = if is_svg && time.is_none() && svg::animated(&data)? {
        Some(0.0)
    } else {
        time
    };
    let pixels = if is_svg {
        if let Some(time) = time {
            animation::render(&path, &data, time)?
        } else {
            svg::render(&data)?
        }
    } else {
        anyhow::ensure!(time.is_none(), "time_seconds is only supported for SVG");
        let mut reader = image::ImageReader::new(Cursor::new(data))
            .with_guessed_format()
            .context("detecting raster image format")?;
        anyhow::ensure!(
            matches!(reader.format(), Some(ImageFormat::Png | ImageFormat::Jpeg)),
            "supported inputs: SVG, PNG, JPEG"
        );
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(8192);
        limits.max_image_height = Some(8192);
        limits.max_alloc = Some(128 * 1024 * 1024);
        reader.limits(limits);
        let image = reader.decode().context("decoding raster image")?;
        anyhow::ensure!(
            u64::from(image.width()) * u64::from(image.height()) <= MAX_PIXELS,
            "raster image exceeds 16 megapixels"
        );
        image.thumbnail(MAX_SIDE, MAX_SIDE).to_rgba8()
    };
    let (width, height) = pixels.dimensions();
    // Composite onto white so transparent SVG backgrounds have predictable contrast.
    let mut opaque = RgbaImage::new(width, height);
    for (source, target) in pixels.pixels().zip(opaque.pixels_mut()) {
        let alpha = u32::from(source[3]);
        for c in 0..3 {
            target[c] = ((u32::from(source[c]) * alpha + 255 * (255 - alpha) + 127) / 255) as u8;
        }
        target[3] = 255;
    }
    let mut png = Cursor::new(Vec::new());
    opaque
        .write_to(&mut png, ImageFormat::Png)
        .context("encoding image tool PNG")?;
    Ok(Picture {
        png: png.into_inner(),
        width,
        height,
        source: path.display().to_string(),
        svg: is_svg,
        time_seconds: time,
    })
}

impl Picture {
    pub fn tool_result(&self) -> Value {
        json!({"content":[
            {"type":"text","text":format!("{} · {}×{} pixels · {}", self.source, self.width, self.height,
                match self.time_seconds { Some(t) => format!("SVG frame sampled {t}s after load; no desktop capture"),
                    None if self.svg => "static SVG rendered in Rust".into(), None => "raster image".into() })},
            {"type":"image","mimeType":"image/png","data":STANDARD.encode(&self.png)}
        ],"isError":false})
    }
}

pub fn read_cli(input: &str, output: Option<&str>, time: Option<f64>) -> Result<()> {
    let picture = read_at(input, time)?;
    if let Some(output) = output {
        let target = Path::new(output);
        anyhow::ensure!(
            target
                .extension()
                .is_some_and(|x| x.eq_ignore_ascii_case("png")),
            "output must have a .png extension"
        );
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(target)
            .context("creating PNG output (existing files are not overwritten)")?;
        file.write_all(&picture.png).context("writing PNG output")?;
        println!(
            "{}",
            json!({"path":target.canonicalize()?.display().to_string(),"width":picture.width,"height":picture.height,"mimeType":"image/png"})
        );
    } else {
        println!("{}", picture.tool_result());
    }
    Ok(())
}
