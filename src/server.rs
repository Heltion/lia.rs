//! Serial local chat endpoint; model execution remains in the Rust CUDA engine.
use crate::{
    gguf::{Model, Value},
    inference::{Engine, Snapshot},
    tokenizer::Tokenizer,
};
use anyhow::{Result, anyhow};
use serde_json::{Value as Json, json};
use std::time::Instant;
struct Session {
    engine: Engine,
    tokens: Vec<u32>,
    checkpoint: Option<Snapshot>,
}

mod session;

struct Job {
    path: String,
    input: Json,
    reply: tokio::sync::oneshot::Sender<Result<Json>>,
}

fn worker(
    path: String,
    capacity: usize,
    ready: std::sync::mpsc::SyncSender<Result<()>>,
    mut jobs: tokio::sync::mpsc::UnboundedReceiver<Job>,
) {
    let loaded = (|| -> Result<_> {
        let model = Model::open(&path)?;
        let tokenizer = Tokenizer::from_model(&model)?;
        let engine = Engine::load_from_model(&path, capacity, &model)?;
        Ok((
            model,
            tokenizer,
            Session {
                engine,
                tokens: Vec::new(),
                checkpoint: None,
            },
        ))
    })();
    let (model, tokenizer, mut session) = match loaded {
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
            "/health" => Ok(
                json!({"status":"ok","backend":"lia.rs","thinking":true,"mtp":true,"mtp_max_draft":session.engine.max_draft(),"target_verify_max_batch":session.engine.max_draft()+1,"mtp_state_restore":"captured per-position recurrent states","prefill_batch":512,"prefill_precision":"f16","prefill_small_batch":8,"prefill_tensor_min_tokens":9,"prefill_dense_min_tokens":crate::gpu::DENSE_MIN_BATCH,"prefill_wide_min_tokens":crate::gpu::WIDE_MIN_BATCH,"kv_cache":"f16","activation_quantization":session.engine.integer_enabled(),"matrix_split_k":session.engine.matrix_layout().0,"matrix_row_group":session.engine.matrix_layout().1,"matrix_block_threads":session.engine.matrix_threads(),"attention":"partitioned_online_softmax","attention_tile":128,"attention_decode_tile":64}),
            ),
            "/v1/chat/completions" => session.chat(&model, &tokenizer, &job.input),
            "/apply-template" => {
                crate::chat::render(&model, &job.input).map(|prompt| json!({"prompt":prompt}))
            }
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
        let _ = job.reply.send(response);
    }
}

async fn handle(
    axum::extract::State(jobs): axum::extract::State<tokio::sync::mpsc::UnboundedSender<Job>>,
    uri: axum::http::Uri,
    body: axum::body::Bytes,
) -> (axum::http::StatusCode, axum::Json<Json>) {
    let result = async {
        let input = if body.is_empty() {
            json!({})
        } else {
            serde_json::from_slice(&body)?
        };
        let (reply, response) = tokio::sync::oneshot::channel();
        jobs.send(Job {
            path: uri.path().to_owned(),
            input,
            reply,
        })
        .map_err(|_| anyhow!("inference worker stopped"))?;
        response
            .await
            .map_err(|_| anyhow!("inference worker stopped"))?
    }
    .await;
    match result {
        Ok(value) => (axum::http::StatusCode::OK, axum::Json(value)),
        Err(error) => (
            axum::http::StatusCode::BAD_REQUEST,
            axum::Json(json!({"error":{"message":format!("{error:#}")}})),
        ),
    }
}

pub fn run(path: &str, port: u16, capacity: usize) -> Result<()> {
    use anyhow::Context;
    let (jobs, receiver) = tokio::sync::mpsc::unbounded_channel();
    let (ready, loaded) = std::sync::mpsc::sync_channel(1);
    let path = path.to_owned();
    let thread = std::thread::Builder::new()
        .name("inference".into())
        .spawn(move || worker(path, capacity, ready, receiver))?;
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
        eprintln!("lia.rs listening on 127.0.0.1:{port}; thinking enabled; built-in MTP enabled");
        axum::serve(listener, app).await.context("serving HTTP")
    });
    thread
        .join()
        .map_err(|_| anyhow!("inference worker panicked"))?;
    result
}
