#![cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};
use lia::{
    inference::{Engine, duplex::Conversation},
    models::minicpm_o45::{audio::units::PcmUnits, duplex::Control, resample::Resampler, wav},
    sampling::{duplex, speech},
    tokenizer::Tokenizer,
};
use serde_json::{Value, json};
use std::{path::Path, time::Instant};
#[test]
#[ignore = "requires official checkpoint, input speech WAV and CUDA"]
fn continuous_native_audio_language_and_speech_session() -> Result<()> {
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let input = std::env::var("LIA_MINICPMO_CONVERSATION_WAV")?;
    let result = std::env::var("LIA_MINICPMO_CONVERSATION_RESULT")?;
    let started = Instant::now();
    let tokenizer = Tokenizer::from_huggingface(&model)?;
    let config: Value = serde_json::from_reader(std::fs::File::open(
        Path::new(&model).join("tokenizer_config.json"),
    )?)?;
    let mut forbidden: Vec<u32> = serde_json::from_value(config["bad_token_ids"].clone())?;
    forbidden.push(151722);
    let mut engine = Engine::load_huggingface(&model, 2048, false)?;
    engine.load_audio(&model)?;
    engine.load_speech(&model)?;
    engine.load_codec_encoder(&model)?;
    engine.load_codec_flow(&model)?;
    engine.load_codec_hift(&model)?;
    engine.load_reference_voice_weights(&model)?;
    let voice = engine.prepare_voice_wav(
        &std::fs::read(Path::new(&model).join("assets/HT_ref_audio.wav"))?,
        6.,
    )?;
    let prefix = tokenizer.encode(
        "<|im_start|>system\nStreaming Omni Conversation.<|im_end|>",
        true,
    )?;
    let make_sampler = || {
        duplex::Sampler::new(
            duplex::Parameters {
                greedy: true,
                ..Default::default()
            },
            forbidden.clone(),
            Vec::new(),
            42,
        )
    };
    let mut conversation = Conversation::new(
        &mut engine,
        &prefix,
        Control::new(20, 0, true)?,
        make_sampler()?,
        speech::Sampler::new(
            speech::Parameters {
                policy: speech::Policy::Vllm,
                temperature: 0.8,
                top_k: 100,
                top_p: 0.8,
                repetition_penalty: 1.05,
                min_tokens: 0,
            },
            42,
        )?,
        voice,
        42,
    )?;
    let preparation_seconds = started.elapsed().as_secs_f64();
    let audio = wav::decode(&std::fs::read(&input)?)?;
    let mut samples = Resampler::new(audio.sample_rate, 16000)?.process(&audio.samples)?;
    ensure!(
        samples.len() <= 160000,
        "conversation smoke input exceeds ten seconds"
    );
    samples.extend(vec![0.; 160000]);
    let mut pending = PcmUnits::default();
    let mut units = Vec::new();
    let mut pcm16 = Vec::new();
    let mut transcript = String::new();
    let mut first_audio = None;
    let run = Instant::now();
    for fragment in samples.chunks(3200) {
        pending.append(fragment)?;
        while let Some(input) = pending.next(false) {
            let begin = Instant::now();
            let output =
                conversation.audio_unit(&mut engine, &tokenizer, &input.samples, || false)?;
            if !output.audio.pcm16.is_empty() && first_audio.is_none() {
                first_audio = Some(run.elapsed().as_secs_f64());
            }
            println!(
                "unit {}: {:?}, text={:?}, speech_codes={}, samples={}",
                units.len(),
                output.language.decision_tokens,
                output.text,
                output.speech_codes,
                output.audio.speech_samples
            );
            units.push(json!({"decisions":output.language.decision_tokens,
                "text":output.text,"end_of_turn":output.language.end_of_turn,
                "is_listen":output.language.is_listen,"speech_codes":output.speech_codes,
                "speech_sampled_tokens":output.speech_sampled_tokens,
                "speech_samples":output.audio.speech_samples,"padding_samples":output.audio.padding_samples,
                "seconds":begin.elapsed().as_secs_f64(),"position":output.language.end_position}));
            transcript.push_str(&output.text);
            pcm16.extend(output.audio.pcm16);
        }
    }
    conversation.cancel();
    let position = engine.position();
    ensure!(
        !conversation.valid(),
        "cancel did not invalidate conversation"
    );
    ensure!(
        conversation
            .audio_unit(&mut engine, &tokenizer, &[0.; 16000], || false)
            .is_err()
            && engine.position() == position,
        "cancelled conversation changed language cache"
    );
    let record = json!({"scope":"native continuous-session smoke; greedy language, stochastic speech; not reference acceptance",
        "input":input,"preparation_seconds":preparation_seconds,"client_seconds":run.elapsed().as_secs_f64(),
        "first_audio_seconds":first_audio,"transcript":transcript,"units":units,
        "cancel_preserved_position":position,"speech_output_bytes":pcm16.len()});
    let root = Path::new(&result);
    std::fs::create_dir_all(root)?;
    std::fs::write(
        root.join("result.json"),
        serde_json::to_vec_pretty(&record)?,
    )?;
    let samples: Vec<f32> = pcm16
        .chunks_exact(2)
        .map(|v| i16::from_le_bytes([v[0], v[1]]) as f32 / 32768.)
        .collect();
    std::fs::write(root.join("audio.wav"), wav::pcm16(&samples, 24000)?)?;
    ensure!(!units.is_empty(), "no input unit executed");
    ensure!(
        !transcript.is_empty() && !pcm16.is_empty(),
        "native conversation produced no spoken response; see {}",
        root.display()
    );
    println!(
        "{}",
        serde_json::to_string(&record).context("encoding conversation results")?
    );
    Ok(())
}
