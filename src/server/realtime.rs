//! Bounded WebSocket transport; all model state stays on the inference worker.
use super::*;
use axum::extract::{State, WebSocketUpgrade, ws::Message};
use futures_util::{SinkExt, StreamExt};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
mod config;
mod events;
mod input;
mod worker;
pub(super) use worker::run;
pub(super) struct Channels {
    input: tokio::sync::mpsc::Receiver<Command>,
    pub(super) output: tokio::sync::mpsc::Sender<Json>,
    disconnected: Arc<AtomicBool>,
    epoch: Arc<AtomicU64>,
    ga: bool,
}
struct Command {
    event: Json,
    epoch: u64,
}
struct Disconnect(Arc<AtomicBool>);
impl Drop for Disconnect {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}
pub(super) async fn upgrade(
    State(jobs): State<tokio::sync::mpsc::UnboundedSender<Job>>,
    headers: axum::http::HeaderMap,
    ws: WebSocketUpgrade,
) -> axum::response::Response {
    let ga = !headers
        .get("openai-beta")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.split(',').any(|v| v.trim() == "realtime=v1"));
    ws.max_message_size(16 * 1024 * 1024).on_upgrade(move |socket| async move {
        let (mut sink, mut source) = socket.split();
        let (send, input) = tokio::sync::mpsc::channel(32);
        let (output, mut receive) = tokio::sync::mpsc::channel::<Json>(32);
        let disconnected = Arc::new(AtomicBool::new(false));
        let epoch = Arc::new(AtomicU64::new(0));
        let _guard = Disconnect(disconnected.clone());
        let writer_disconnected = disconnected.clone();
        let writer = tokio::spawn(async move {
            while let Some(event) = receive.recv().await {
                if sink.send(Message::Text(event.to_string().into())).await.is_err() { break; }
            }
            writer_disconnected.store(true, Ordering::Release);
            let _ = sink.close().await;
        });
        let (reply, _) = tokio::sync::oneshot::channel();
        if jobs.send(Job {
            submitted: Instant::now(), path: "/v1/realtime".into(), input: Json::Null,
            reply, stream: None,
            realtime: Some(Channels {input, output: output.clone(), disconnected: disconnected.clone(), epoch: epoch.clone(), ga}),
        }).is_err() {
            let _ = output.send(json!({"type":"error","error":{"message":"inference worker stopped"}})).await;
        } else {
            while let Some(message) = source.next().await {
                let event = match message {
                    Ok(Message::Text(text)) => serde_json::from_str::<Json>(&text),
                    Ok(Message::Close(_)) | Err(_) => break,
                    _ => continue,
                };
                match event {
                    Ok(event) => {
                        if matches!(event["type"].as_str(), Some("response.cancel" | "barge_in" | "output_audio_buffer.clear")) {
                            epoch.fetch_add(1, Ordering::AcqRel);
                        }
                        let close = event["type"] == "session.close";
                        let command = Command { event, epoch: epoch.load(Ordering::Acquire) };
                        if send.send(command).await.is_err() || close { break; }
                    }
                    Err(error) => {
                        if output.send(json!({"type":"error","error":{"message":format!("invalid realtime JSON: {error}")}})).await.is_err() { break; }
                    }
                }
            }
        }
        disconnected.store(true, Ordering::Release);
        drop(send);
        drop(output);
        let _ = writer.await;
    })
}
