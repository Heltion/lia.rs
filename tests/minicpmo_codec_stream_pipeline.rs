#![cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> Result<(PathBuf, Value)> {
    let root = PathBuf::from(std::env::var(name).with_context(|| format!("read {name}"))?);
    let info = serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    Ok((root, info))
}
fn read(root: &Path, file: &Value) -> Result<Vec<f32>> {
    let path = root.join(file.as_str().context("missing streaming tensor filename")?);
    let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    ensure!(bytes.len().is_multiple_of(4), "invalid Float32 tensor");
    Ok(bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
fn max_error(actual: &[f32], expected: &[f32]) -> Result<f32> {
    ensure!(
        actual.len() == expected.len(),
        "streaming tensor extent mismatch"
    );
    ensure!(
        actual.iter().chain(expected).all(|v| v.is_finite()),
        "nonfinite streaming tensor"
    );
    Ok(actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max))
}

/// Propagate native encoder errors through CFM and HiFT instead of resetting each
/// stage to reference intermediates. Explicit reference noise isolates arithmetic.
#[test]
#[ignore = "requires official streaming fixtures, MiniCPM-o checkpoint and CUDA"]
fn native_codes_to_streamed_wave_match_official_pipeline() -> Result<()> {
    let (encoder_root, encoder_info) = fixture("LIA_MINICPMO_CODEC_STREAM_ORACLE")?;
    let (flow_root, flow_info) = fixture("LIA_MINICPMO_CODEC_FLOW_STREAM_ORACLE")?;
    let (hift_root, hift_info) = fixture("LIA_MINICPMO_CODEC_HIFT_STREAM_ORACLE")?;
    let encoder = &encoder_info["groups"][0];
    let flow = &flow_info["groups"][0];
    let chunks = encoder["chunks"]
        .as_array()
        .context("missing encoder chunks")?;
    let flows = flow["chunks"].as_array().context("missing flow chunks")?;
    let waves = hift_info["chunks"]
        .as_array()
        .context("missing waveform chunks")?;
    ensure!(
        chunks.len() == flows.len() && chunks.len() == waves.len() + 1,
        "inconsistent pipeline fixtures"
    );
    let prefix = encoder["prefix_frames"]
        .as_u64()
        .context("missing prefix")? as usize;
    ensure!(
        flow["prefix_frames"].as_u64() == Some(prefix as u64),
        "inconsistent prompt prefix"
    );
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let mut engine = lia::inference::Engine::load_huggingface(&model, 64, false)?;
    engine.load_codec_encoder(&model)?;
    engine.load_codec_flow(&model)?;
    engine.load_codec_hift(&model)?;
    let mut encoder_cache = engine.new_codec_encoder_cache()?;
    let mut flow_cache = engine.new_codec_flow_cache(10)?;
    let mut hift_cache = engine.new_codec_hift_cache()?;
    let mut combined = Vec::new();
    for (index, (chunk, flow)) in chunks.iter().zip(flows).enumerate() {
        let codes: Vec<u32> = serde_json::from_value(chunk["codes"].clone())?;
        let last = chunk["last"].as_bool().context("missing final flag")?;
        let mu = engine.codec_encoder_chunk(&mut encoder_cache, &codes, last)?;
        let encoder_error = max_error(&mu, &read(&encoder_root, &chunk["features"])?)?;
        ensure!(encoder_error < 0.01, "coupled encoder mismatch");
        let mel = engine.codec_flow_chunk(
            &mut flow_cache,
            &read(&flow_root, &flow["noise"])?,
            &mu,
            &read(&flow_root, &flow["speaker"])?,
            &read(&flow_root, &flow["condition"])?,
        )?;
        let mel_error = max_error(&mel, &read(&flow_root, &flow["mel"])?)?;
        ensure!(mel_error < 0.02, "coupled flow mismatch");
        engine.trim_codec_encoder_cache(&mut encoder_cache, prefix, 100)?;
        engine.trim_codec_flow_cache(&mut flow_cache, prefix, 100)?;
        if index == 0 {
            ensure!(mel.len() == prefix * 80, "prompt mel extent mismatch");
            continue;
        }
        let chunk = &waves[index - 1];
        let wave = engine.codec_hift_chunk(
            &mut hift_cache,
            &mel,
            &read(&hift_root, &chunk["noise"])?,
            last,
        )?;
        let expected = read(&hift_root, &chunk["wave"])?;
        max_error(&wave, &expected)?;
        let (mut error, mut energy, mut actual_energy, mut dot) = (0., 0., 0., 0.);
        for (&a, &b) in wave.iter().zip(&expected) {
            let (a, b) = (f64::from(a), f64::from(b));
            error += (a - b).powi(2);
            energy += b * b;
            actual_energy += a * a;
            dot += a * b;
        }
        ensure!(
            energy > 0. && actual_energy > 0.,
            "silent pipeline waveform"
        );
        let relative = (error / energy).sqrt();
        let cosine = dot / (energy * actual_energy).sqrt();
        println!(
            "chunk {index}: encoder_max={encoder_error}, mel_max={mel_error}, samples={}, relative_l2={relative}, cosine={cosine}",
            wave.len()
        );
        ensure!(relative < 0.5 && cosine > 0.9, "coupled waveform mismatch");
        combined.extend(wave);
    }
    ensure!(
        combined.len() == 75840 && hift_cache.finished() && encoder_cache.finished(),
        "incomplete streaming pipeline"
    );
    std::fs::write(
        hift_root.join("native-pipeline-stream.wav"),
        lia::models::minicpm_o45::wav::pcm16(&combined, 24000)?,
    )?;
    let (voice_root, voice_info) = fixture("LIA_MINICPMO_VOICE_ORACLE")?;
    let voice_info = &voice_info["cases"][0];
    let prompt: Vec<u32> = serde_json::from_value(voice_info["prompt_codes"].clone())?;
    let prompt_mel = read(&voice_root, &voice_info["prompt_mel"])?;
    let embedding = read(&voice_root, &voice_info["speaker"])?;
    let voice = lia::models::minicpm_o45::Voice::new(&prompt, &prompt_mel, &embedding)?;
    let mut controller =
        engine.new_codec_voice_stream(&voice, 10, &read(&flow_root, &flows[0]["noise"])?)?;
    let mut controller_wave = Vec::new();
    ensure!(
        engine
            .codec_voice_stream_chunk(&mut controller, &[6561], &[], &[], true)
            .is_err()
            && controller.generated_samples() == 0,
        "invalid controller input advanced state"
    );
    for index in 1..chunks.len() {
        let codes: Vec<u32> = serde_json::from_value(chunks[index]["codes"].clone())?;
        let wave = engine.codec_voice_stream_chunk(
            &mut controller,
            &codes,
            &read(&flow_root, &flows[index]["noise"])?,
            &read(&hift_root, &waves[index - 1]["noise"])?,
            index + 1 == chunks.len(),
        )?;
        let expected = read(&hift_root, &waves[index - 1]["wave"])?;
        max_error(&wave, &expected)?;
        let error = wave
            .iter()
            .zip(&expected)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum::<f64>();
        let energy = expected.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
        let actual_energy = wave.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
        let dot = wave
            .iter()
            .zip(&expected)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum::<f64>();
        let relative = (error / energy).sqrt();
        let cosine = dot / (energy * actual_energy).sqrt();
        println!(
            "controller chunk {index}: samples={}, relative_l2={relative}, cosine={cosine}, history_frames={}",
            wave.len(),
            controller.history_frames()
        );
        ensure!(
            relative < 0.5 && cosine > 0.9 && controller.history_frames() <= prefix + 100,
            "streaming controller mismatch"
        );
        controller_wave.extend(wave);
    }
    ensure!(
        controller.finished()
            && controller.generated_samples() == 75840
            && engine
                .codec_voice_stream_chunk(&mut controller, &[1], &[], &[], true)
                .is_err(),
        "controller lifecycle mismatch"
    );
    std::fs::write(
        hift_root.join("native-controller-stream.wav"),
        lia::models::minicpm_o45::wav::pcm16(&controller_wave, 24000)?,
    )?;
    Ok(())
}
