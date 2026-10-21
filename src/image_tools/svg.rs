//! Static, self-contained SVG rasterization through resvg's Rust renderer.
use super::{MAX_SIDE, RgbaImage};
use anyhow::{Context, Result};
use resvg::{tiny_skia, usvg};

pub(super) fn animated(data: &[u8]) -> Result<bool> {
    let text = std::str::from_utf8(data).context("SVG must be UTF-8")?;
    let document = roxmltree::Document::parse(text).context("parsing SVG animation metadata")?;
    Ok(document.descendants().any(|node| {
        node.is_element()
            && (matches!(
                node.tag_name().name(),
                "animate" | "animateMotion" | "animateTransform" | "set"
            ) || node.attributes().any(|attr| {
                attr.name() == "style" && attr.value().to_ascii_lowercase().contains("animation")
            }) || node.tag_name().name() == "style"
                && node
                    .text()
                    .unwrap_or("")
                    .to_ascii_lowercase()
                    .contains("@keyframes"))
    }))
}

pub(super) fn render(data: &[u8]) -> Result<RgbaImage> {
    let text = std::str::from_utf8(data).context("SVG must be UTF-8")?;
    let document = roxmltree::Document::parse(text).context("parsing SVG XML")?;
    let root = document.root_element();
    anyhow::ensure!(
        root.tag_name().name() == "svg"
            && root.tag_name().namespace() == Some("http://www.w3.org/2000/svg"),
        "expected an SVG document"
    );
    for node in root.descendants().filter(|n| n.is_element()) {
        let name = node.tag_name().name();
        anyhow::ensure!(
            !matches!(
                name,
                "animate"
                    | "animateMotion"
                    | "animateTransform"
                    | "set"
                    | "script"
                    | "foreignObject"
            ),
            "static SVG only: unsupported element <{name}>; pass time_seconds to read_image for an animated SVG frame"
        );
        for attr in node.attributes() {
            if attr.name() == "href" {
                anyhow::ensure!(
                    attr.value().starts_with('#') || attr.value().starts_with("data:"),
                    "SVG must be self-contained; external href is unsupported"
                );
            }
            if attr.name() == "style" {
                check_style(attr.value())?;
            }
        }
        if name == "style" {
            check_style(node.text().unwrap_or(""))?;
        }
    }
    let mut options = usvg::Options::default();
    // Never fetch filesystem images or network resources through an SVG reference.
    options.image_href_resolver.resolve_string = Box::new(|_, _| None);
    options.fontdb_mut().load_system_fonts();
    let tree = usvg::Tree::from_data(data, &options).context("resolving static SVG")?;
    let size = tree.size();
    let scale = (MAX_SIDE as f32 / size.width().max(size.height())).min(1.0);
    let width = (size.width() * scale).ceil().max(1.0) as u32;
    let height = (size.height() * scale).ceil().max(1.0) as u32;
    let mut pixmap = tiny_skia::Pixmap::new(width, height).context("allocating SVG raster")?;
    resvg::render(
        &tree,
        tiny_skia::Transform::from_scale(scale, scale),
        &mut pixmap.as_mut(),
    );
    // tiny-skia stores premultiplied RGBA; image expects straight RGBA.
    let mut pixels = pixmap.take();
    for pixel in pixels.chunks_exact_mut(4) {
        let alpha = u32::from(pixel[3]);
        if alpha != 0 {
            for value in &mut pixel[..3] {
                *value = ((u32::from(*value) * 255 + alpha / 2) / alpha).min(255) as u8;
            }
        }
    }
    RgbaImage::from_raw(width, height, pixels).context("converting SVG pixels")
}

fn check_style(style: &str) -> Result<()> {
    let style = style.to_ascii_lowercase();
    anyhow::ensure!(
        !style.contains("animation") && !style.contains("@keyframes") && !style.contains("@import"),
        "static SVG only: CSS animation and imports are unsupported"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn renders_shape_pixels_and_preserves_alpha() -> Result<()> {
        let image = render(br##"<svg xmlns="http://www.w3.org/2000/svg" width="120" height="80"><rect width="40" height="80" fill="#ff0000"/><circle cx="80" cy="40" r="20" fill="#0000ff" fill-opacity="0.5"/></svg>"##)?;
        assert_eq!(image.dimensions(), (120, 80));
        assert_eq!(image.get_pixel(20, 40).0, [255, 0, 0, 255]);
        assert_eq!(&image.get_pixel(80, 40).0[..3], &[0, 0, 255]);
        assert!((127..=128).contains(&image.get_pixel(80, 40).0[3]));
        assert_eq!(image.get_pixel(115, 40).0[3], 0);
        Ok(())
    }
    #[test]
    fn rejects_animation_external_resources_and_non_svg() {
        for body in [
            "<animate/>",
            "<script/>",
            "<image href='file:///etc/passwd'/>",
            "<style>@keyframes spin{}</style>",
        ] {
            let source = format!("<svg xmlns='http://www.w3.org/2000/svg'>{body}</svg>");
            assert!(render(source.as_bytes()).is_err(), "{body}");
        }
        assert!(render(b"<html/>").is_err());
    }
}
