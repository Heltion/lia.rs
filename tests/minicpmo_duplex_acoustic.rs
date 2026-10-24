#![cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};
use lia::{
    models::minicpm_o45::{
        Voice,
        duplex::audio::{Schedule, packet},
    },
    sampling::speech::{Parameters, Policy, Sampler},
};
use serde_json::Value;
use std::path::Path;
fn read(root: &Path, name: &Value) -> Result<Vec<f32>> {
    let bytes =
        std::fs::read(root.join(name.as_str().context("missing acoustic fixture filename")?))?;
    ensure!(
        bytes.len().is_multiple_of(4),
        "truncated acoustic Float32 tensor"
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
fn compare(actual: &[f32], expected: &[f32]) -> Result<(f64, f64)> {
    ensure!(
        actual.len() == expected.len() && actual.iter().chain(expected).all(|v| v.is_finite()),
        "acoustic waveform extent/value mismatch"
    );
    let (mut error, mut energy, mut norm, mut dot) = (0., 0., 0., 0.);
    for (&a, &b) in actual.iter().zip(expected) {
        let (a, b) = (f64::from(a), f64::from(b));
        error += (a - b).powi(2);
        energy += b * b;
        norm += a * a;
        dot += a * b;
    }
    ensure!(energy > 0. && norm > 0., "silent acoustic reference/result");
    let relative = (error / energy).sqrt();
    let cosine = dot / (energy * norm).sqrt();
    ensure!(
        relative < 0.5 && cosine > 0.9,
        "duplex acoustic mismatch: relative={relative}, cosine={cosine}"
    );
    Ok((relative, cosine))
}
#[test]
#[ignore = "requires official duplex TTS/acoustic fixtures, checkpoint and CUDA"]
fn retained_tts_to_acoustic_units_match_official_flush_and_padding() -> Result<()> {
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_DUPLEX_ACOUSTIC_ORACLE")?);
    let info: Value = serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let speech_root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_SPEECH_CHUNK_ORACLE")?);
    let speech: Value =
        serde_json::from_reader(std::fs::File::open(speech_root.join("info.json"))?)?;
    let voice_root = Path::new(
        info["voice_oracle"]
            .as_str()
            .context("missing reference voice")?,
    );
    let voice: Value = serde_json::from_reader(std::fs::File::open(voice_root.join("info.json"))?)?;
    let voice = &voice["cases"][0];
    let prompt: Vec<u32> = serde_json::from_value(voice["prompt_codes"].clone())?;
    let mel = read(voice_root, &voice["prompt_mel"])?;
    let embedding = read(voice_root, &voice["speaker"])?;
    let voice = Voice::new(&prompt, &mel, &embedding)?;
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let mut engine = lia::inference::Engine::load_huggingface(&model, 64, false)?;
    engine.load_speech(&model)?;
    engine.load_codec_encoder(&model)?;
    engine.load_codec_flow(&model)?;
    engine.load_codec_hift(&model)?;
    let mut acoustic =
        engine.new_codec_voice_stream(&voice, 10, &read(&root, &info["prompt_noise"])?)?;
    let mut speech_cache = engine.new_speech_cache(512)?;
    let mut sampler = Sampler::new(
        Parameters {
            policy: Policy::Chunk,
            min_tokens: 0,
            ..Parameters::default()
        },
        42,
    )?;
    let mut schedule = Schedule::default();
    let mut block_index = 0;
    let mut pcm = Vec::new();
    let mut speech_samples = 0;
    let mut padded = 0;
    let chunks = speech["groups"][0]["chunks"]
        .as_array()
        .context("missing spoken units")?;
    for (index, unit) in chunks.iter().enumerate() {
        let tokens: Vec<u32> = serde_json::from_value(unit["tokens"].clone())?;
        let conditioning =
            engine.duplex_speech_conditioning(&tokens, &read(&speech_root, &unit["hidden"])?)?;
        sampler
            .set_min_tokens(unit["minimum"].as_u64().context("missing speech minimum")? as usize)?;
        let generated =
            engine.speech_generate_chunk(&mut speech_cache, &conditioning, &mut sampler, 26)?;
        ensure!(
            serde_json::to_value(&generated.codes)? == unit["codes"],
            "native speech codes changed before acoustic streaming"
        );
        let last = index + 1 == chunks.len();
        let blocks = schedule.push(&generated.codes, last, index == 0)?;
        let expected_unit = &info["units"][index];
        ensure!(
            blocks.len() as u64
                == expected_unit["chunks"]
                    .as_u64()
                    .context("missing chunk count")?,
            "duplex scheduling count mismatch"
        );
        let mut wave = Vec::new();
        for block in blocks {
            let expected = &info["chunks"][block_index];
            ensure!(
                serde_json::to_value(&block.codes)? == expected["codes"]
                    && expected["last"] == block.last
                    && expected["unit"].as_u64() == Some(index as u64),
                "duplex lookahead/flush boundary mismatch"
            );
            let actual = engine.codec_voice_stream_chunk(
                &mut acoustic,
                &block.codes,
                &read(&root, &expected["flow_noise"])?,
                &read(&root, &expected["hift_noise"])?,
                block.last,
            )?;
            let (relative, cosine) = compare(&actual, &read(&root, &expected["wave"])?)?;
            println!(
                "block {block_index}: samples={}, relative_l2={relative}, cosine={cosine}",
                actual.len()
            );
            wave.extend(actual);
            block_index += 1;
        }
        let audio = packet(&wave, last)?;
        ensure!(
            expected_unit["speech_samples"].as_u64() == Some(audio.speech_samples as u64)
                && expected_unit["padding_samples"].as_u64() == Some(audio.padding_samples as u64)
                && expected_unit["buffered_codes"].as_u64()
                    == Some(schedule.buffered_codes() as u64),
            "duplex packet padding/buffering mismatch"
        );
        let samples: Vec<f32> = audio
            .pcm16
            .chunks_exact(2)
            .map(|v| f32::from(i16::from_le_bytes(v.try_into().unwrap())) / 32768.)
            .collect();
        let (relative, cosine) = compare(&samples, &read(&root, &expected_unit["wave"])?)?;
        println!(
            "unit {index}: speech_samples={}, padding_samples={}, relative_l2={relative}, cosine={cosine}",
            audio.speech_samples, audio.padding_samples
        );
        speech_samples += audio.speech_samples;
        padded += audio.padding_samples;
        pcm.extend(samples);
    }
    ensure!(
        block_index == 5
            && schedule.finished()
            && acoustic.finished()
            && speech_samples
                == info["speech_samples"]
                    .as_u64()
                    .context("missing total speech")? as usize
            && padded
                == info["padding_samples"]
                    .as_u64()
                    .context("missing total padding")? as usize,
        "duplex acoustic stream did not finish completely"
    );
    std::fs::write(
        root.join("native-duplex-units.wav"),
        lia::models::minicpm_o45::wav::pcm16(&pcm, 24000)?,
    )?;
    Ok(())
}
