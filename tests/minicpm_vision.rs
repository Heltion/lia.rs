#![cfg(feature = "cuda")]

use anyhow::{Context, Result};
use lia::inference::Engine;

#[test]
#[ignore = "requires local MiniCPM checkpoint, image and official vision feature reference"]
fn checkpoint_features_match_official_vision() -> Result<()> {
    let model = std::env::var("LIA_TEST_MODEL").context("set LIA_TEST_MODEL")?;
    let image = std::env::var("LIA_TEST_IMAGE").context("set LIA_TEST_IMAGE")?;
    let reference =
        std::env::var("LIA_TEST_VISION_REFERENCE").context("set LIA_TEST_VISION_REFERENCE")?;
    let pixels = std::fs::read(&image).with_context(|| format!("reading {image}"))?;
    let prepared = lia::models::minicpm_v47::prepare(&pixels, 1)?;
    let mut engine = Engine::load_huggingface(&model, 1024, true)?;
    let actual = engine.image_embeddings(&prepared.slices[0])?;
    if let Ok(path) = std::env::var("LIA_TEST_VISION_ACTUAL") {
        let bytes: Vec<u8> = actual
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(&path, bytes).context("writing external vision feature diagnostic")?;
    }
    let bytes = std::fs::read(&reference).with_context(|| format!("reading {reference}"))?;
    anyhow::ensure!(
        bytes.len() == actual.len() * 4,
        "vision feature count differs"
    );
    let expected: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    anyhow::ensure!(
        actual.iter().all(|x| x.is_finite()),
        "nonfinite vision features"
    );
    let dot = actual
        .iter()
        .zip(&expected)
        .map(|(&a, &b)| a as f64 * b as f64)
        .sum::<f64>();
    let norm = |v: &[f32]| v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
    let error = actual
        .iter()
        .zip(&expected)
        .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
        .sum::<f64>();
    let cosine = dot / (norm(&actual) * norm(&expected)).sqrt();
    let relative = (error / norm(&expected)).sqrt();
    let bf16 = std::env::var("LIA_MINICPM_VISION_BF16").as_deref() == Ok("1");
    if bf16 {
        anyhow::ensure!(
            actual.iter().all(|value| value.to_bits() & 0xffff == 0),
            "vision output does not preserve BF16 activation boundaries"
        );
    }
    eprintln!(
        "MiniCPM vision elements={} cosine={cosine} relative_l2={relative}",
        actual.len()
    );
    anyhow::ensure!(
        cosine > 0.99 && relative < 0.05,
        "vision feature deviation exceeds diagnostic tolerance"
    );
    Ok(())
}

#[test]
#[ignore = "child-process helper for resident_and_offloaded_multislice_logits_match"]
fn multislice_logits_snapshot() -> Result<()> {
    let model = std::env::var("LIA_TEST_MODEL").context("set LIA_TEST_MODEL")?;
    let image = std::env::var("LIA_TEST_IMAGE").context("set LIA_TEST_IMAGE")?;
    let output = std::env::var("LIA_TEST_LOGITS_PATH").context("set LIA_TEST_LOGITS_PATH")?;
    let pixels = std::fs::read(&image).with_context(|| format!("reading {image}"))?;
    let mut images = Vec::new();
    let mut length = 0;
    for _ in 0..2 {
        let prepared = lia::models::minicpm_v47::prepare(&pixels, 1)?;
        let slice = prepared
            .slices
            .into_iter()
            .next()
            .context("missing image slice")?;
        let count = slice.width / 56 * (slice.height / 56);
        images.push((length, slice));
        length += count;
    }
    let mut tokens = vec![1; length];
    tokens.push(2);
    let mut plan = lia::vision::PositionPlan::new();
    plan.append_text(tokens.len())?;
    let mut engine = Engine::load_huggingface(&model, tokens.len().max(1024), true)?;
    engine.prepare_image_inputs(&images, plan)?;
    let mut offset = 0;
    while offset < tokens.len() {
        let count = engine.prefill_chunk(tokens.len() - offset)?;
        engine.forward_many(&tokens[offset..offset + count], false)?;
        offset += count;
    }
    let bytes: Vec<u8> = engine
        .current_logits()?
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    std::fs::write(&output, bytes).with_context(|| format!("writing {output}"))?;
    Ok(())
}

#[test]
#[ignore = "requires local MiniCPM checkpoint and image; starts two serial native GPU processes"]
fn resident_and_offloaded_multislice_logits_match() -> Result<()> {
    let executable = std::env::current_exe().context("locating integration test executable")?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .context("reading temporary directory timestamp")?
        .as_nanos();
    let directory = std::env::temp_dir().join(format!("lia-vision-residency-{stamp}"));
    std::fs::create_dir(&directory).context("creating temporary vision comparison directory")?;
    let mut snapshots = Vec::new();
    for offload in ["0", "1"] {
        let path = directory.join(format!("{offload}.f32"));
        let status = std::process::Command::new(&executable)
            .args([
                "--ignored",
                "--exact",
                "multislice_logits_snapshot",
                "--nocapture",
            ])
            .env("LIA_VISION_OFFLOAD", offload)
            .env("LIA_TEST_LOGITS_PATH", &path)
            .status()
            .context("running native vision residency comparison")?;
        anyhow::ensure!(status.success(), "vision residency child failed: {status}");
        snapshots.push(std::fs::read(&path).context("reading native vision snapshot")?);
    }
    anyhow::ensure!(!snapshots[0].is_empty(), "empty native vision snapshot");
    anyhow::ensure!(
        snapshots[0] == snapshots[1],
        "vision residency changes multislice logits"
    );
    eprintln!(
        "Resident/offloaded multislice logits bitwise identical: {} bytes",
        snapshots[0].len()
    );
    std::fs::remove_dir_all(&directory).context("removing temporary vision snapshots")?;
    Ok(())
}
