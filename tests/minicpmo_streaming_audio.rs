use anyhow::{Context, Result, ensure};
use lia::models::minicpm_o45::audio::{Features, stream::Streaming};
use serde_json::Value;
use std::path::Path;

fn read(root: &Path, name: &Value) -> Result<Vec<f32>> {
    let path = root.join(name.as_str().context("missing streaming audio tensor")?);
    let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    ensure!(bytes.len().is_multiple_of(4), "truncated Float32 tensor");
    Ok(bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
fn compare(actual: &Features, reference: &[f32]) -> Result<(f32, f64)> {
    ensure!(
        actual.values.len() == reference.len() && reference.len() == actual.frames * 80,
        "streaming mel extent mismatch"
    );
    ensure!(
        actual.values.iter().chain(reference).all(|v| v.is_finite()),
        "nonfinite streaming mel"
    );
    let maximum = actual
        .values
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max);
    let rms = (actual
        .values
        .iter()
        .zip(reference)
        .map(|(a, b)| f64::from(a - b).powi(2))
        .sum::<f64>()
        / reference.len().max(1) as f64)
        .sqrt();
    Ok((maximum, rms))
}
fn field(info: &Value, name: &str) -> Result<usize> {
    Ok(info[name]
        .as_u64()
        .with_context(|| format!("missing streaming field {name}"))? as usize)
}
#[test]
#[ignore = "requires external official StreamingMelProcessorExact fixtures"]
fn streaming_mel_matches_official_emission_normalization_and_sliding() -> Result<()> {
    verify(false)
}
#[test]
#[ignore = "requires external official StreamingMelProcessorExact NumPy fixtures"]
fn streaming_mel_matches_official_numpy_path() -> Result<()> {
    verify(true)
}
fn verify(numpy: bool) -> Result<()> {
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_STREAMING_MEL_ORACLE")?);
    let info: Value = serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let mut failures = 0;
    let (max_gate, rms_gate) = if numpy { (1e-5, 1e-6) } else { (5e-4, 5e-5) };
    for case in info["cases"]
        .as_array()
        .context("missing streaming cases")?
    {
        let mut stream = Streaming::new(1000, 1035, 20, case["sliding"] == true)?;
        for (index, chunk) in case["chunks"]
            .as_array()
            .context("missing streaming chunks")?
            .iter()
            .enumerate()
        {
            let samples = read(&root, &chunk["input"])?;
            if case["name"] != "short-final" || index == 0 {
                ensure!(
                    stream.requested_samples() == field(chunk, "requested_samples")?,
                    "first-hop alignment mismatch"
                );
            }
            let mut snapshot = if index == 5 || index == 29 {
                Some(stream.clone())
            } else {
                None
            };
            let actual = stream.push(&samples, chunk["last"] == true)?;
            let (maximum, rms) = compare(
                &actual.features,
                &read(&root, &chunk[if numpy { "numpy_mel" } else { "mel" }])?,
            )?;
            failures += usize::from(maximum >= max_gate || rms >= rms_gate);
            let expected = &chunk["info"];
            ensure!(
                actual.features.frames == field(expected, "emitted_frames")?
                    && actual.stable_frames == field(expected, "stable_T")?
                    && actual.full_frames == field(expected, "T_full")?
                    && actual.core_start == field(expected, "core_start")?
                    && actual.core_end == field(expected, "core_end")?
                    && actual.buffer_samples == field(expected, "buffer_len_samples")?
                    && stream.state().base_frame == field(expected, "base_T")?
                    && stream.state().dropped_samples == field(expected, "left_samples_dropped")?
                    && stream.state().chunks == field(expected, "chunk_number")?,
                "streaming boundary metadata mismatch"
            );
            if let Some(snapshot) = &mut snapshot {
                let replay = snapshot.push(&samples, chunk["last"] == true)?;
                ensure!(
                    replay.features.values == actual.features.values
                        && snapshot.state() == stream.state(),
                    "streaming snapshot restore mismatch"
                );
            }
            println!(
                "{} chunk {index}: frames={}, max_abs={maximum}, rms={rms}, base={}",
                case["name"],
                actual.features.frames,
                stream.state().base_frame
            );
        }
        let tail = stream.flush()?;
        let (maximum, rms) = compare(
            &tail,
            &read(&root, &case[if numpy { "numpy_tail" } else { "tail" }])?,
        )?;
        failures += usize::from(maximum >= max_gate || rms >= rms_gate);
        ensure!(
            tail.frames == field(case, "tail_frames")? && stream.flush()?.frames == 0,
            "streaming tail missing or repeated"
        );
        let state = &case["final_state"];
        ensure!(
            stream.state().processed_samples == field(state, "total_samples_processed")?
                && stream.state().emitted_frames == field(state, "last_emitted_T")?
                && stream.buffer_samples() == field(state, "buffer_len")?,
            "final streaming state mismatch"
        );
    }
    ensure!(
        failures == 0,
        "{failures} streaming mel comparisons exceed precision gates (max {max_gate}, rms {rms_gate}); boundary metadata checks passed"
    );
    Ok(())
}
