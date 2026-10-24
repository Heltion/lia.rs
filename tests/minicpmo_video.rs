#![cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};
use lia::{inference::Engine, models::minicpm_o45::prepare, tokenizer::Tokenizer};
use serde_json::Value;
use std::path::Path;
fn read(path: impl AsRef<Path>) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    ensure!(
        bytes.len().is_multiple_of(4),
        "truncated video oracle tensor"
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
#[test]
#[ignore = "requires official weighted video-frame oracle, checkpoint and CUDA"]
fn native_frame_markers_slices_and_parked_language_cache_match_reference() -> Result<()> {
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_VIDEO_ORACLE")?);
    let info: Value = serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let tokenizer = Tokenizer::from_huggingface(&model)?;
    let mut engine = Engine::load_huggingface(&model, 512, true)?;
    engine.forward_many(&[1, 2], false)?;
    let saved = engine.snapshot()?;
    let expected_logits = engine.forward(3)?;
    engine.restore(&saved)?;
    ensure!(
        engine.duplex_frame_embeddings(&[], &tokenizer).is_err() && engine.position() == 2,
        "invalid frame group changed language history"
    );
    let cases = info["frames"].as_array().context("missing video frames")?;
    let frames = cases
        .iter()
        .map(|case| {
            let image = case["image"].as_str().context("missing frame image")?;
            prepare(
                &std::fs::read(image)?,
                case["max_slices"].as_u64().context("missing slice count")? as usize,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        frames[0].slices.len() > 1 && frames[1].slices.len() == 1,
        "video oracle must exercise both HD and single-slice frames"
    );
    ensure!(
        engine
            .duplex_frame_embeddings(&frames[..1], &tokenizer)
            .is_err()
            && engine.position() == 2,
        "standalone video frame accepted unsupported HD policy"
    );
    let output = engine.duplex_frame_embeddings(&frames, &tokenizer)?;
    ensure!(
        engine.position() == 2,
        "frame encoding advanced language KV"
    );
    let markers = read(root.join("markers.f32"))?;
    let mut offset = 0;
    let mut maximum = 0.0f32;
    for case in cases {
        for (index, file) in case["features"]
            .as_array()
            .context("missing slice features")?
            .iter()
            .enumerate()
        {
            let marker = if index == 0 { 0 } else { 2 };
            ensure!(
                output[offset..offset + 4096] == markers[marker * 4096..(marker + 1) * 4096],
                "native frame opener embedding differs from checkpoint"
            );
            offset += 4096;
            let expected = read(root.join(file.as_str().context("missing slice tensor")?))?;
            ensure!(
                expected.len() == 64 * 4096,
                "incorrect official vision rows"
            );
            let actual = &output[offset..offset + expected.len()];
            let mut squared = 0.0f64;
            let mut slice_max = 0.0f32;
            for (&a, &b) in actual.iter().zip(&expected) {
                ensure!(a.is_finite() && b.is_finite(), "nonfinite frame embedding");
                let error = a - b;
                slice_max = slice_max.max(error.abs());
                squared += f64::from(error).powi(2);
            }
            let rms = (squared / expected.len() as f64).sqrt();
            ensure!(
                slice_max < 0.05 && rms < 0.01,
                "video slice exceeds existing vision precision gate"
            );
            maximum = maximum.max(slice_max);
            offset += expected.len();
            ensure!(
                output[offset..offset + 4096] == markers[(marker + 1) * 4096..(marker + 2) * 4096],
                "native frame closer embedding differs from checkpoint"
            );
            offset += 4096;
        }
    }
    ensure!(offset == output.len(), "extra or missing video frame rows");
    let actual_logits = engine.forward(3)?;
    ensure!(
        actual_logits == expected_logits,
        "vision residency changed language cache or weights"
    );
    println!(
        "native video: frames={}, rows={}, image_max_abs={maximum}, marker_max_abs=0, language_logits_bitexact=true",
        cases.len(),
        output.len() / 4096
    );
    Ok(())
}
