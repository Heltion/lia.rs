//! Keep speech-code and waveform timings distinct from language token throughput.
use super::*;
use base64::{Engine as _, engine::general_purpose::STANDARD};
impl State {
    pub(in crate::server) fn finish(
        &self,
        engine: &Engine,
        tokenizer: &Tokenizer,
        capture: Capture,
        message: &mut Json,
        output: Option<&stream::Output>,
    ) -> Result<Json> {
        anyhow::ensure!(
            capture.collector.is_none(),
            "spoken text lacks its closing TTS boundary; increase max_tokens"
        );
        anyhow::ensure!(
            !capture.segments.is_empty(),
            "model did not produce spoken text"
        );
        let voice = self
            .reference
            .as_ref()
            .context("missing prepared voice")?
            .1
            .voice()?;
        let start = Instant::now();
        let mut waveform = Vec::new();
        let mut sampled = 0;
        let mut code_count = 0;
        let mut conditioning_count = 0;
        let mut prefill_ms = 0.;
        let mut decode_ms = 0.;
        let mut waveform_ms = 0.;
        let mut first_audio_ms = None;
        let mut first_audio_code_n = None;
        for (index, segment) in capture.segments.iter().enumerate() {
            let conditioning = engine.speech_conditioning(&segment.tokens, &segment.hidden)?;
            let mut renderer = output
                .map(|_| {
                    streaming::Render::new(
                        engine,
                        &voice,
                        capture.options.seed.wrapping_add(index as u64),
                        waveform.len(),
                    )
                })
                .transpose()?;
            let generated = engine.speech_generate_with(
                &conditioning,
                capture.options.parameters,
                capture.options.maximum,
                capture.options.seed.wrapping_add(index as u64),
                |code| {
                    anyhow::ensure!(
                        !output.is_some_and(|o| o.is_closed()),
                        "stream client disconnected during speech"
                    );
                    if let (Some(renderer), Some(output)) = (&mut renderer, output) {
                        renderer.push(
                            engine,
                            code,
                            output,
                            &capture.options.format,
                            capture.started,
                        )?;
                    }
                    Ok(())
                },
            )?;
            anyhow::ensure!(
                generated.finished,
                "speech decoder reached audio max_tokens before EOS"
            );
            sampled += generated.sampled_tokens;
            code_count += generated.codes.len();
            conditioning_count += generated.conditioning_tokens;
            prefill_ms += generated.prefill_seconds * 1000.;
            decode_ms += generated.decode_seconds * 1000.;
            if let (Some(mut renderer), Some(output)) = (renderer, output) {
                renderer.finish(engine, output, &capture.options.format, capture.started)?;
                if first_audio_ms.is_none() {
                    first_audio_ms = renderer.first_audio_ms;
                    first_audio_code_n = renderer.first_audio_code_n;
                }
                waveform_ms += renderer.waveform_ms;
                waveform.extend(renderer.wave);
            } else {
                let render = Instant::now();
                waveform.extend(engine.codec_render(
                    &generated.codes,
                    &voice,
                    10,
                    capture.options.seed.wrapping_add(index as u64),
                )?);
                waveform_ms += render.elapsed().as_secs_f64() * 1000.;
            }
        }
        let duration = waveform.len() as f64 / 24000.;
        let synthesis_ms = start.elapsed().as_secs_f64() * 1000.;
        let wav = crate::models::minicpm_o45::wav::pcm16(&waveform, 24000)?;
        let bytes = if capture.options.format == "pcm16" {
            &wav[44..]
        } else {
            &wav
        };
        let transcript = capture
            .segments
            .iter()
            .map(|segment| {
                Ok(String::from_utf8(tokenizer.decode_text(&segment.tokens)?)?
                    .trim()
                    .to_owned())
            })
            .collect::<Result<Vec<_>>>()?
            .join("\n");
        let audio_id = output
            .map(stream::Output::audio_id)
            .unwrap_or_else(|| format!("audio-{}", super::super::stream::completion_id()));
        message["audio"] = json!({"id":audio_id,"data":STANDARD.encode(bytes),"expires_at":super::super::stream::unix_seconds()+3600,"transcript":transcript,"format":capture.options.format,"sample_rate":24000});
        if !capture.options.text {
            message["content"] = Json::Null;
        }
        Ok(
            json!({"setup_ms":capture.setup_ms,"conditioning_n":conditioning_count,"sampled_code_n":sampled,"code_n":code_count,"speech_prefill_ms":prefill_ms,"speech_decode_ms":decode_ms,"waveform_ms":waveform_ms,"synthesis_ms":synthesis_ms,"audio_seconds":duration,"synthesis_rtf":synthesis_ms/1000./duration,"first_audio_ms":first_audio_ms.unwrap_or_else(|| capture.started.elapsed().as_secs_f64()*1000.),"first_audio_code_n":first_audio_code_n,"audio_delivery":if output.is_some() { "incremental" } else { "buffered" },"sampling":{"policy":format!("{:?}",capture.options.parameters.policy),"temperature":capture.options.parameters.temperature,"top_k":capture.options.parameters.top_k,"top_p":capture.options.parameters.top_p,"repetition_penalty":capture.options.parameters.repetition_penalty,"seed":capture.options.seed}}),
        )
    }
}
