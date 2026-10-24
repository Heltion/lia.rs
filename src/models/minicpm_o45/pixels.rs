//! MiniCPM-o RGB preprocessing; model inference remains on the CUDA path.
use super::images::ImagePlan;
use crate::vision::PreparedImage;
use anyhow::Result;

pub struct Prepared {
    pub plan: ImagePlan,
    pub slices: Vec<PreparedImage>,
}

fn normalized(image: &image::RgbImage) -> PreparedImage {
    let (width, height) = (image.width() as usize, image.height() as usize);
    let count = width * height;
    let mut pixels = vec![0.0; count * 3];
    for channel in 0..3 {
        for i in 0..count {
            let rescaled = (image.as_raw()[i * 3 + channel] as f64 / 255.0) as f32;
            pixels[channel * count + i] = (rescaled - 0.5) / 0.5;
        }
    }
    PreparedImage {
        width,
        height,
        pixels,
    }
}

fn resized(image: &image::RgbImage, [width, height]: [usize; 2]) -> image::RgbImage {
    let bytes = crate::vision::resize::bicubic(
        image.as_raw(),
        image.width() as usize,
        image.height() as usize,
        width,
        height,
    );
    // The shared resampler writes exactly width*height RGB triplets.
    image::RgbImage::from_raw(width as u32, height as u32, bytes).unwrap()
}

pub fn prepare(bytes: &[u8], max_slices: usize) -> Result<Prepared> {
    let decoded = crate::vision::decode(bytes)?;
    let plan = ImagePlan::new(
        decoded.width() as usize,
        decoded.height() as usize,
        max_slices,
    )?;
    let mut slices = vec![normalized(&resized(&decoded, plan.thumbnail))];
    if let (Some([columns, rows]), Some(size)) = (plan.grid, plan.refined) {
        let refined = resized(&decoded, size);
        let (width, height) = (size[0] / columns, size[1] / rows);
        for row in 0..rows {
            for column in 0..columns {
                let slice = image::imageops::crop_imm(
                    &refined,
                    (column * width) as u32,
                    (row * height) as u32,
                    width as u32,
                    height as u32,
                )
                .to_image();
                slices.push(normalized(&slice));
            }
        }
    }
    Ok(Prepared { plan, slices })
}
