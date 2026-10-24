#![cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};
use std::path::Path;
fn floats(root: &Path, file: &serde_json::Value) -> Result<Vec<f32>> {
    let bytes = std::fs::read(root.join(file.as_str().context("missing acoustic fixture path")?))?;
    ensure!(
        bytes.len().is_multiple_of(4),
        "truncated Float32 acoustic fixture"
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
#[test]
#[ignore = "requires official voice/acoustic oracle, checkpoint and CUDA"]
fn conditioned_codes_to_mel_and_waveform_match_reference() -> Result<()> {
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_VOICE_ORACLE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let mut engine = lia::inference::Engine::load_huggingface(&model, 128, false)?;
    engine.load_codec_encoder(&model)?;
    engine.load_codec_flow(&model)?;
    engine.load_codec_hift(&model)?;
    for case in info["cases"].as_array().context("missing acoustic cases")? {
        let codes: Vec<u32> = serde_json::from_value(case["codes"].clone())?;
        let prompt: Vec<u32> = serde_json::from_value(case["prompt_codes"].clone())?;
        let prompt_mel = floats(&root, &case["prompt_mel"])?;
        let embedding = floats(&root, &case["speaker"])?;
        let voice = lia::models::minicpm_o45::Voice::new(&prompt, &prompt_mel, &embedding)?;
        let mel = engine.codec_voice_mel(&codes, &voice, &floats(&root, &case["noise"])?, 10)?;
        let expected = floats(&root, &case["mel"])?;
        ensure!(
            mel.len() == codes.len() * 2 * 80 && mel.len() == expected.len(),
            "conditioned mel cropping mismatch"
        );
        let mut maximum = 0.0f32;
        let mut squared = 0.0f64;
        for (&a, &b) in mel.iter().zip(&expected) {
            ensure!(a.is_finite() && b.is_finite(), "nonfinite conditioned mel");
            maximum = maximum.max((a - b).abs());
            squared += f64::from(a - b).powi(2);
        }
        let rms = (squared / mel.len() as f64).sqrt();
        println!("{} mel: max_abs={maximum}, rms={rms}", case["name"]);
        ensure!(
            maximum < 0.02 && rms < 0.004,
            "conditioned mel reference mismatch"
        );
        let wave = engine.codec_synthesize_noise(&mel, &floats(&root, &case["hift_noise"])?)?;
        let expected = floats(&root, &case["wave"])?;
        ensure!(
            wave.len() == codes.len() * 2 * 480
                && wave.len() == expected.len()
                && wave.iter().all(|v| v.is_finite() && v.abs() <= 0.99),
            "invalid rendered waveform"
        );
        let mut error = 0.0;
        let mut energy = 0.0;
        let mut actual_energy = 0.0;
        let mut dot = 0.0;
        for (&a, &b) in wave.iter().zip(&expected) {
            let (a, b) = (f64::from(a), f64::from(b));
            error += (a - b).powi(2);
            energy += b * b;
            actual_energy += a * a;
            dot += a * b;
        }
        let relative = (error / energy).sqrt();
        let cosine = dot / (energy * actual_energy).sqrt();
        println!(
            "{} waveform: samples={}, relative_l2={relative}, cosine={cosine}",
            case["name"],
            wave.len()
        );
        ensure!(
            relative < 0.5 && cosine > 0.9,
            "rendered waveform reference mismatch"
        );
        let wav = lia::models::minicpm_o45::wav::pcm16(&wave, 24000)?;
        std::fs::write(
            root.join(format!("{}-native.wav", case["name"].as_str().unwrap())),
            wav,
        )?;
    }
    Ok(())
}

#[test]
#[ignore = "requires official voice/acoustic oracle, reference WAV, checkpoint and CUDA"]
fn native_reference_wav_to_waveform_matches_reference() -> Result<()> {
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_VOICE_ORACLE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let case = info["cases"]
        .as_array()
        .context("missing acoustic cases")?
        .iter()
        .find(|c| c["name"] == "reference-prompt")
        .context("missing reference prompt case")?;
    let codes: Vec<u32> = serde_json::from_value(case["codes"].clone())?;
    let mut engine = lia::inference::Engine::load_huggingface(&model, 128, false)?;
    engine.load_reference_voice_weights(&model)?;
    engine.load_codec_encoder(&model)?;
    engine.load_codec_flow(&model)?;
    engine.load_codec_hift(&model)?;
    let prepared = engine.prepare_voice_wav(
        &std::fs::read(Path::new(&model).join("assets/HT_ref_audio.wav"))?,
        6.,
    )?;
    let voice = prepared.voice()?;
    let prompt: Vec<u32> = serde_json::from_value(case["prompt_codes"].clone())?;
    ensure!(
        voice.codes() == prompt,
        "native reference preprocessing changed prompt codes"
    );
    let mel = engine.codec_voice_mel(&codes, &voice, &floats(&root, &case["noise"])?, 10)?;
    let expected = floats(&root, &case["mel"])?;
    ensure!(
        mel.len() == expected.len(),
        "native voice mel extent mismatch"
    );
    let maximum = mel
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max);
    let rms = (mel
        .iter()
        .zip(&expected)
        .map(|(a, b)| f64::from(a - b).powi(2))
        .sum::<f64>()
        / mel.len() as f64)
        .sqrt();
    println!("native WAV conditioning mel: max_abs={maximum}, rms={rms}");
    ensure!(
        maximum < 0.02 && rms < 0.004,
        "native voice conditioned mel reference mismatch"
    );
    let wave = engine.codec_synthesize_noise(&mel, &floats(&root, &case["hift_noise"])?)?;
    let expected = floats(&root, &case["wave"])?;
    ensure!(
        wave.len() == expected.len()
            && wave.len() == codes.len() * 960
            && wave.iter().all(|v| v.is_finite() && v.abs() <= 0.99),
        "invalid native voice waveform"
    );
    let mut error = 0.;
    let mut energy = 0.;
    let mut actual_energy = 0.;
    let mut dot = 0.;
    for (&a, &b) in wave.iter().zip(&expected) {
        let (a, b) = (f64::from(a), f64::from(b));
        error += (a - b).powi(2);
        energy += b * b;
        actual_energy += a * a;
        dot += a * b;
    }
    let relative = (error / energy).sqrt();
    let cosine = dot / (energy * actual_energy).sqrt();
    println!(
        "native WAV conditioning waveform: samples={}, relative_l2={relative}, cosine={cosine}",
        wave.len()
    );
    ensure!(
        relative < 0.5 && cosine > 0.9,
        "native voice waveform reference mismatch"
    );
    std::fs::write(
        root.join("native-reference-wav-native.wav"),
        lia::models::minicpm_o45::wav::pcm16(&wave, 24000)?,
    )?;
    Ok(())
}
