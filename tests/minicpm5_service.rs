#![cfg(all(feature = "cuda", feature = "image-tools"))]
use anyhow::{Context, Result, ensure};
use reqwest::blocking::Client;
use serde_json::{Value, json};
use std::{
    process::{Child, Command, Stdio},
    sync::{Arc, Barrier},
    time::{Duration, Instant},
};

struct Server {
    child: Child,
    _cache: tempfile::TempDir,
    base: String,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Server {
    fn start(host_mib: usize, client: &Client) -> Result<Self> {
        let model = std::env::var("LIA_TEST_MINICPM5_MODEL")?;
        let binary = std::env::var("LIA_TEST_MINICPM5_BINARY")
            .unwrap_or_else(|_| env!("CARGO_BIN_EXE_lia").to_owned());
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        drop(listener);
        let cache = tempfile::tempdir().context("creating service-test KV directory")?;
        let child = Command::new(binary)
            .args([
                "serve",
                &model,
                &port.to_string(),
                "1024",
                "--batch",
                "--max-sequences",
                "2",
                "--kv-pool-tokens",
                "1024",
                "--kv-host-mib",
                &host_mib.to_string(),
                "--kv-ssd-mib",
                "64",
                "--kv-cache-dir",
            ])
            .arg(cache.path())
            .env("LIA_CUDA_GRAPH", "1")
            .env("LIA_PREALLOCATE", "1")
            .env("LIA_PREFILL_BATCH", "1024")
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .context("starting MiniCPM5 service")?;
        let mut server = Self {
            child,
            _cache: cache,
            base: format!("http://127.0.0.1:{port}"),
        };
        let deadline = Instant::now() + Duration::from_secs(120);
        while Instant::now() < deadline {
            ensure!(
                server.child.try_wait()?.is_none(),
                "MiniCPM5 service exited before readiness"
            );
            if client
                .get(format!("{}/health", server.base))
                .send()
                .is_ok_and(|response| response.status().is_success())
            {
                return Ok(server);
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        anyhow::bail!("MiniCPM5 service readiness timed out")
    }
}

fn request(base: &str, client: &Client, key: usize) -> Result<Value> {
    let mut archive = String::from("Archive of access codes:\n");
    for index in 0..32 {
        archive.push_str(&format!(
            "Record {index:02}: code {}; retained for auditing.\n",
            100000 + index * 27183
        ));
    }
    archive.push_str(&format!(
        "\nWhat is the access code for Record {key:02}? Reply with its six digits."
    ));
    let response: Value = client
        .post(format!("{base}/v1/chat/completions"))
        .timeout(Duration::from_secs(120))
        .json(
            &json!({"model":"minicpm5","messages":[{"role":"user","content":archive}],
            "temperature":0,"max_tokens":1024,"chat_template_kwargs":{"enable_thinking":true}}),
        )
        .send()
        .context("requesting concurrent MiniCPM5 completion")?
        .error_for_status()
        .context("MiniCPM5 completion status")?
        .json()
        .context("decoding MiniCPM5 completion")?;
    let choice = &response["choices"][0];
    let expected = (100000 + key * 27183).to_string();
    ensure!(
        choice["finish_reason"] == "stop"
            && choice["message"]["content"]
                .as_str()
                .is_some_and(|text| text.contains(&expected)),
        "incorrect or incomplete resumed request {key}: {response}"
    );
    Ok(response)
}

#[test]
#[ignore = "requires MiniCPM5 checkpoint, CUDA, and a native service binary"]
fn active_requests_resume_from_ram_and_ssd_under_kv_pressure() -> Result<()> {
    let client = Client::builder().timeout(Duration::from_secs(5)).build()?;
    for host_mib in [64, 0] {
        let server = Server::start(host_mib, &client)?;
        let barrier = Arc::new(Barrier::new(3));
        std::thread::scope(|scope| -> Result<()> {
            let handles: Vec<_> = [17, 11]
                .into_iter()
                .map(|key| {
                    let barrier = Arc::clone(&barrier);
                    let client = &client;
                    let base = &server.base;
                    scope.spawn(move || {
                        barrier.wait();
                        request(base, client, key)
                    })
                })
                .collect();
            barrier.wait();
            for handle in handles {
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("service-test client panicked"))??;
            }
            Ok(())
        })?;
        let health: Value = client
            .get(format!("{}/health", server.base))
            .send()?
            .error_for_status()?
            .json()?;
        let counter = if host_mib == 0 {
            "ssd_cache_hits"
        } else {
            "ram_cache_hits"
        };
        ensure!(
            health["kv_preemptions"].as_u64().unwrap_or(0) > 0,
            "KV pressure did not pause a request: {health}"
        );
        ensure!(
            health[counter].as_u64().unwrap_or(0) > 0,
            "paused request did not restore from {counter}: {health}"
        );
        ensure!(
            health["active_requests"] == 0 && health["paused_requests"] == 0,
            "completed requests retained scheduler slots: {health}"
        );
        println!("KV-pressure host_mib={host_mib} passed: {health}");
    }
    Ok(())
}
