//! Serial local chat endpoint; model execution remains in the Rust CUDA engine.
use crate::{
    gguf::{Model, Value},
    inference::{Engine, Snapshot},
    tokenizer::Tokenizer,
};
use anyhow::{Context, Result, anyhow};
use serde_json::{Value as Json, json};
use std::time::Instant;
struct Session {
    template: String,
    eos_tokens: Vec<u32>,
    engine: Engine,
    tokens: Vec<u32>,
    checkpoint: Snapshot,
    processor: Option<crate::models::image::Processor>,
    image_sources: Vec<String>,
}

mod images;
mod sampling;
mod session;
mod stream;

struct Job {
    path: String,
    input: Json,
    reply: tokio::sync::oneshot::Sender<Result<Json>>,
    stream: Option<stream::Output>,
}

fn worker(
    path: String,
    projector: Option<String>,
    capacity: usize,
    ready: std::sync::mpsc::SyncSender<Result<()>>,
    mut jobs: tokio::sync::mpsc::UnboundedReceiver<Job>,
) {
    let loaded = (|| -> Result<_> {
        let (tokenizer, engine, processor, template, eos_tokens) = if std::path::Path::new(&path)
            .is_dir()
        {
            anyhow::ensure!(
                projector.is_none(),
                "Hugging Face checkpoints include their own vision weights"
            );
            let config: Json = serde_json::from_reader(std::fs::File::open(
                std::path::Path::new(&path).join("generation_config.json"),
            )?)?;
            let eos_tokens = config["eos_token_id"]
                .as_array()
                .ok_or_else(|| anyhow!("missing Hugging Face EOS token list"))?
                .iter()
                .map(|v| {
                    Ok(u32::try_from(
                        v.as_u64().ok_or_else(|| anyhow!("invalid EOS token"))?,
                    )?)
                })
                .collect::<Result<Vec<_>>>()?;
            let vision = std::env::var("LIA_TEXT_ONLY").as_deref() != Ok("1");
            let engine = Engine::load_huggingface(&path, capacity, vision)?;
            anyhow::ensure!(
                !eos_tokens.is_empty()
                    && eos_tokens
                        .iter()
                        .all(|id| (*id as usize) < engine.vocabulary_size()),
                "invalid EOS tokens"
            );
            let processor = if vision {
                let (minimum, maximum) = images::token_limits(&engine);
                Some(crate::models::image::Processor::from_huggingface(
                    &path, minimum, maximum,
                )?)
            } else {
                None
            };
            (
                Tokenizer::from_huggingface(&path)?,
                engine,
                processor,
                std::fs::read_to_string(std::path::Path::new(&path).join("chat_template.jinja"))?,
                eos_tokens,
            )
        } else {
            let model = Model::open(&path)?;
            let tokenizer = Tokenizer::from_model(&model)?;
            let template = match model.metadata.get("tokenizer.chat_template") {
                Some(Value::String(s)) => s.clone(),
                _ => return Err(anyhow!("missing chat template")),
            };
            let eos = match model.metadata.get("tokenizer.ggml.eos_token_id") {
                Some(Value::Unsigned(id)) => u32::try_from(*id)?,
                _ => return Err(anyhow!("missing EOS token")),
            };
            let (engine, processor) = if let Some(projector) = &projector {
                let vision = Model::open(projector)?;
                let engine = Engine::load_with_vision(&path, capacity, projector)?;
                let (minimum, maximum) = images::token_limits(&engine);
                let processor = crate::vision::Processor::from_model(&vision, minimum, maximum)?;
                (
                    engine,
                    Some(crate::models::image::Processor::Qwen(processor)),
                )
            } else {
                (Engine::load_from_model(&path, capacity, &model)?, None)
            };
            (tokenizer, engine, processor, template, vec![eos])
        };
        let checkpoint = engine
            .prepare_checkpoint()
            .context("allocating service checkpoint")?;
        Ok((
            tokenizer,
            Session {
                template,
                eos_tokens,
                engine,
                tokens: Vec::new(),
                checkpoint,
                processor,
                image_sources: Vec::new(),
            },
        ))
    })();
    let (tokenizer, mut session) = match loaded {
        Ok(loaded) => loaded,
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    if ready.send(Ok(())).is_err() {
        return;
    }
    while let Some(job) = jobs.blocking_recv() {
        let response = match job.path.as_str() {
            "/health" => {
                let mut info = json!({"status":"ok","backend":"lia.rs","image_input":session.processor.is_some(),"image_token_limits":session.processor.as_ref().and_then(|p| match p { crate::models::image::Processor::Qwen(_) => Some(images::token_limits(&session.engine)), _ => None }),"attention_workspace_bytes":session.engine.attention_reserved_bytes(),"query_tiled_attention":session.engine.query_tiled_attention(),"gdn_parallel_chunk_preparation":session.engine.gdn_parallel(),"cuda_graph":session.engine.cuda_graph_enabled(),"cuda_graph_stats":session.engine.cuda_graph_stats(),"gpu_buffers_preallocated":session.engine.buffers_preallocated(),"scratch_reserved_bytes":session.engine.scratch_reserved_bytes(),"thinking":true,"mtp":session.engine.mtp_enabled(),"sampling_modes":["greedy","top_k_top_p"],"sampling_top_k_max":20,"mtp_max_draft":session.engine.max_draft(),"stochastic_mtp":session.engine.stochastic_mtp(),"target_verify_max_batch":session.engine.max_draft()+1,"mtp_state_restore":"captured per-position recurrent states","prefill_batch":session.engine.prefill_batch(),"prefill_precision":session.engine.prefill_precision(),"prefill_fp8_min_tokens":65,"prefill_fp8_tile":128,"prefill_fp8_weight_workspace_bytes":session.engine.fp8_weight_workspace_bytes(),"prefill_fp8_weight_prepack_min_tokens":crate::gpu::fp8_cache::MIN_BATCH,"prefill_small_batch":8,"prefill_tensor_min_tokens":9,"prefill_medium_min_tokens":crate::gpu::MEDIUM_MIN_BATCH,"prefill_wide_min_tokens":crate::gpu::WIDE_MIN_BATCH,"prefill_wide_token_tile":session.engine.prefill_tile(),"kv_cache":"f16","activation_quantization":session.engine.integer_enabled(),"matrix_split_k":session.engine.matrix_layout().0,"matrix_row_group":session.engine.matrix_layout().1,"matrix_block_threads":session.engine.matrix_threads(),"attention":"partitioned_online_softmax_with_tensor_core_prefill","attention_tile":256,"attention_tensor_min_batch":session.engine.attention_mma_min_batch(),"attention_tensor_decode_min_context":512,"attention_small_batch_tile":128,"attention_decode_tile":64});
                info["host_weight_bytes"] = session.engine.host_weight_bytes().into();
                info["vision_weights_offload"] = session.engine.vision_offload_enabled().into();
                Ok(info)
            }
            "/v1/chat/completions" => session.chat(&tokenizer, &job.input, job.stream.as_ref()),
            "/apply-template" => crate::chat::render_template(&session.template, &job.input)
                .map(|prompt| json!({"prompt":prompt})),
            "/tokenize" => tokenizer
                .encode(
                    job.input
                        .get("content")
                        .and_then(Json::as_str)
                        .unwrap_or(""),
                    job.input
                        .get("parse_special")
                        .and_then(Json::as_bool)
                        .unwrap_or(true),
                )
                .map(|tokens| json!({"tokens":tokens})),
            _ => Err(anyhow!("unknown endpoint")),
        };
        let mut stop = false;
        if response.is_err() && job.path == "/v1/chat/completions" {
            session.tokens.clear();
            session.engine.trim_scratch();
            if let Err(error) = session
                .engine
                .reset()
                .and_then(|()| session.engine.refresh_snapshot(&mut session.checkpoint))
            {
                eprintln!("resetting failed inference session: {error:#}");
                stop = true;
            }
        }
        if let Some(output) = &job.stream {
            output.finish(&response);
        }
        let _ = job.reply.send(response);
        if stop {
            break;
        }
    }
}

async fn handle(
    axum::extract::State(jobs): axum::extract::State<tokio::sync::mpsc::UnboundedSender<Job>>,
    uri: axum::http::Uri,
    body: axum::body::Bytes,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let result = async {
        let input = if body.is_empty() {
            json!({})
        } else {
            serde_json::from_slice(&body)?
        };
        if uri.path() == "/v1/chat/completions"
            && input.get("stream").and_then(Json::as_bool) == Some(true)
        {
            let (output, response) = stream::Output::new(&input);
            let (reply, _) = tokio::sync::oneshot::channel();
            jobs.send(Job {
                path: uri.path().to_owned(),
                input,
                reply,
                stream: Some(output),
            })
            .map_err(|_| anyhow!("inference worker stopped"))?;
            return Ok(response);
        }
        let (reply, response) = tokio::sync::oneshot::channel();
        jobs.send(Job {
            path: uri.path().to_owned(),
            input,
            reply,
            stream: None,
        })
        .map_err(|_| anyhow!("inference worker stopped"))?;
        let value = response
            .await
            .map_err(|_| anyhow!("inference worker stopped"))??;
        Ok::<_, anyhow::Error>(axum::Json(value).into_response())
    }
    .await;
    match result {
        Ok(value) => value,
        Err(error) => (
            axum::http::StatusCode::BAD_REQUEST,
            axum::Json(json!({"error":{"message":format!("{error:#}")}})),
        )
            .into_response(),
    }
}

pub fn run(path: &str, port: u16, capacity: usize) -> Result<()> {
    run_with_projector(path, port, capacity, None)
}

pub fn run_with_projector(
    path: &str,
    port: u16,
    capacity: usize,
    projector: Option<&str>,
) -> Result<()> {
    use anyhow::Context;
    let (jobs, receiver) = tokio::sync::mpsc::unbounded_channel();
    let (ready, loaded) = std::sync::mpsc::sync_channel(1);
    let path = path.to_owned();
    let projector = projector.map(str::to_owned);
    let thread = std::thread::Builder::new()
        .name("inference".into())
        .spawn(move || worker(path, projector, capacity, ready, receiver))?;
    loaded
        .recv()
        .context("waiting for model initialization")??;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let result = runtime.block_on(async {
        let app = axum::Router::new()
            .fallback(handle)
            .layer(axum::extract::DefaultBodyLimit::max(16 * 1024 * 1024))
            .with_state(jobs);
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
        eprintln!("lia.rs listening on 127.0.0.1:{port}; thinking enabled");
        axum::serve(listener, app).await.context("serving HTTP")
    });
    thread
        .join()
        .map_err(|_| anyhow!("inference worker panicked"))?;
    result
}
