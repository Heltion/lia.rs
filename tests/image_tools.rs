#![cfg(feature = "image-tools")]
use anyhow::{Context, Result};
use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use std::{
    io::Write,
    process::{Command, Stdio},
};

#[test]
fn stdio_tool_returns_viewable_pixels_and_reports_missing_files() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let path = temp.path().join("shapes.svg");
    std::fs::write(
        &path,
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="60"><rect width="50" height="60" fill="#ff0000"/><rect x="50" width="50" height="60" fill="#0000ff"/></svg>"##,
    )?;
    let mut child = Command::new(env!("CARGO_BIN_EXE_lia"))
        .arg("image-tools")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    let mut input = child.stdin.take().context("MCP stdin")?;
    for request in [
        json!({"id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}),
        json!({"method":"notifications/initialized"}),
        json!({"id":2,"method":"tools/list"}),
        json!({"id":3,"method":"tools/call","params":{"name":"read_image","arguments":{"path":path}}}),
        json!({"id":4,"method":"tools/call","params":{"name":"read_image","arguments":{"path":temp.path().join("missing.svg")}}}),
    ] {
        writeln!(input, "{request}")?;
    }
    drop(input);
    let output = child.wait_with_output()?;
    anyhow::ensure!(output.status.success(), "MCP server failed");
    let replies = String::from_utf8(output.stdout)?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<Vec<Value>, _>>()?;
    assert_eq!(replies.len(), 4);
    let result = &replies[2]["result"];
    assert_eq!(result["isError"], false);
    let image = result["content"]
        .as_array()
        .context("tool content")?
        .iter()
        .find(|v| v["type"] == "image")
        .context("image content")?;
    let png = STANDARD.decode(image["data"].as_str().context("base64 image")?)?;
    let image = image::load_from_memory(&png)?.to_rgb8();
    assert_eq!(image.dimensions(), (100, 60));
    assert_eq!(image.get_pixel(25, 30).0, [255, 0, 0]);
    assert_eq!(image.get_pixel(75, 30).0, [0, 0, 255]);
    assert_eq!(replies[3]["result"]["isError"], true);
    Ok(())
}

#[test]
#[ignore = "requires local headless Chrome; explicitly run for rendering validation"]
fn animated_svg_frames_follow_requested_time() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let path = temp.path().join("motion.svg");
    std::fs::write(
        &path,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100"><circle cy="50" r="10" fill="red"><animate attributeName="cx" from="20" to="180" dur="4s" repeatCount="indefinite"/></circle></svg>"#,
    )?;
    for (time, expected) in [
        (None, 20.0),
        (Some(0.0), 20.0),
        (Some(1.0), 60.0),
        (Some(2.5), 120.0),
    ] {
        let picture = lia::image_tools::read_at(&path, time)?;
        assert_eq!(picture.time_seconds, Some(time.unwrap_or(0.0)));
        let pixels = image::load_from_memory(&picture.png)?.to_rgb8();
        assert_eq!(pixels.dimensions(), (960, 720));
        let xs: Vec<_> = pixels
            .enumerate_pixels()
            .filter(|(_, _, p)| p[0] > 220 && p[1] < 40 && p[2] < 40)
            .map(|(x, _, _)| f64::from(x))
            .collect();
        anyhow::ensure!(!xs.is_empty(), "missing animated circle");
        let center = xs.iter().sum::<f64>() / xs.len() as f64;
        assert!(
            (center - expected).abs() < 2.0,
            "time={time:?}: center={center}, expected={expected}"
        );
    }
    Ok(())
}
