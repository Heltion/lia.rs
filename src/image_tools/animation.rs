//! Headless Chrome frame capture controlled by Rust CDP; no injected JavaScript.
use super::*;
use std::{
    collections::VecDeque,
    net::TcpStream,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};
use tungstenite::{Message, WebSocket, stream::MaybeTlsStream};

struct Browser {
    child: Child,
    profile: tempfile::TempDir,
}
impl Drop for Browser {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
struct Connection {
    socket: WebSocket<MaybeTlsStream<TcpStream>>,
    id: u64,
    events: VecDeque<Value>,
}
impl Connection {
    fn receive(&mut self) -> Result<Value> {
        loop {
            match self.socket.read().context("reading browser protocol")? {
                Message::Text(text) => {
                    return serde_json::from_str(&text).context("decoding browser protocol");
                }
                Message::Close(_) => anyhow::bail!("headless browser closed unexpectedly"),
                _ => (),
            }
        }
    }
    fn call(&mut self, method: &str, params: Value) -> Result<Value> {
        if std::env::var_os("LIA_BROWSER_TRACE").is_some() {
            eprintln!("CDP {method}");
        }
        self.id += 1;
        let id = self.id;
        self.socket
            .send(Message::text(
                json!({"id":id,"method":method,"params":params}).to_string(),
            ))
            .with_context(|| format!("sending {method}"))?;
        loop {
            let message = self.receive()?;
            if message["id"] == id {
                anyhow::ensure!(
                    message.get("error").is_none(),
                    "{method}: {}",
                    message["error"]
                );
                return Ok(message["result"].clone());
            }
            if message.get("method").is_some() {
                self.events.push_back(message);
            }
            if self.events.len() > 256 {
                self.events.pop_front();
            }
        }
    }
    fn wait(&mut self, name: &str) -> Result<()> {
        if std::env::var_os("LIA_BROWSER_TRACE").is_some() {
            eprintln!("CDP wait {name}");
        }
        if let Some(index) = self.events.iter().position(|v| v["method"] == name) {
            self.events.remove(index);
            return Ok(());
        }
        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline {
            let event = self.receive()?;
            if event["method"] == name {
                return Ok(());
            }
            if event.get("method").is_some() {
                self.events.push_back(event);
            }
            if self.events.len() > 256 {
                self.events.pop_front();
            }
        }
        anyhow::bail!("timeout waiting for {name}")
    }
}

pub(super) fn render(path: &Path, data: &[u8], seconds: f64) -> Result<RgbaImage> {
    anyhow::ensure!(
        seconds.is_finite() && (0.0..=120.0).contains(&seconds),
        "time_seconds must be in 0..=120"
    );
    // Validate before Chrome sees the input. SMIL is allowed; scripts and external resources are not.
    let source = std::str::from_utf8(data).context("SVG must be UTF-8")?;
    let document = roxmltree::Document::parse(source).context("parsing animated SVG")?;
    anyhow::ensure!(
        document.root_element().tag_name().name() == "svg"
            && document.root_element().tag_name().namespace() == Some("http://www.w3.org/2000/svg"),
        "expected SVG document"
    );
    for node in document.descendants().filter(|n| n.is_element()) {
        anyhow::ensure!(
            !matches!(node.tag_name().name(), "script" | "foreignObject"),
            "scripts and foreignObject are unsupported"
        );
        for attr in node.attributes() {
            anyhow::ensure!(
                !attr.name().starts_with("on"),
                "SVG event handlers are unsupported"
            );
            if attr.name() == "href" {
                anyhow::ensure!(
                    attr.value().starts_with('#') || attr.value().starts_with("data:"),
                    "external SVG resources are unsupported"
                );
            }
        }
    }
    let profile = tempfile::tempdir().context("creating isolated browser profile")?;
    let child = Command::new(
        std::env::var_os("LIA_BROWSER").unwrap_or_else(|| "/usr/bin/google-chrome".into()),
    )
    .args([
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-background-networking",
        "--run-all-compositor-stages-before-draw",
        "--no-first-run",
        "--no-default-browser-check",
        "--remote-debugging-port=0",
    ])
    .arg(format!("--user-data-dir={}", profile.path().display()))
    .arg("about:blank")
    .stdin(Stdio::null())
    .stdout(Stdio::null())
    .stderr(Stdio::null())
    .spawn()
    .context("starting headless Chrome")?;
    let mut browser = Browser { child, profile };
    let deadline = Instant::now() + Duration::from_secs(10);
    let port = loop {
        anyhow::ensure!(
            browser.child.try_wait()?.is_none(),
            "headless Chrome exited before readiness"
        );
        if let Ok(text) = std::fs::read_to_string(browser.profile.path().join("DevToolsActivePort"))
        {
            if let Some(port) = text.lines().next() {
                break port.parse::<u16>().context("reading browser port")?;
            }
        }
        anyhow::ensure!(
            Instant::now() < deadline,
            "headless browser readiness timeout"
        );
        std::thread::sleep(Duration::from_millis(20));
    };
    let http = reqwest::blocking::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(5))
        .build()?;
    let targets: Value = http
        .get(format!("http://127.0.0.1:{port}/json/list"))
        .send()?
        .error_for_status()?
        .json()?;
    let url = targets
        .as_array()
        .context("browser target list")?
        .iter()
        .find(|v| v["type"] == "page")
        .and_then(|v| v["webSocketDebuggerUrl"].as_str())
        .context("browser page websocket")?;
    let (mut socket, _) = tungstenite::connect(url).context("connecting browser protocol")?;
    if let MaybeTlsStream::Plain(stream) = socket.get_mut() {
        stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    }
    let mut c = Connection {
        socket,
        id: 0,
        events: VecDeque::new(),
    };
    c.call("Page.enable", json!({}))?;
    c.call(
        "Emulation.setDeviceMetricsOverride",
        json!({"width":960,"height":720,"deviceScaleFactor":1,"mobile":false}),
    )?;
    c.call(
        "Emulation.setScriptExecutionDisabled",
        json!({"value":true}),
    )?;
    c.call("Network.enable", json!({}))?;
    c.call(
        "Network.setBlockedURLs",
        json!({"urls":["http://*","https://*","ftp://*"]}),
    )?;
    let url =
        reqwest::Url::from_file_path(path).map_err(|_| anyhow::anyhow!("invalid SVG file path"))?;
    let navigation = c.call("Page.navigate", json!({"url":url.as_str()}))?;
    anyhow::ensure!(
        navigation.get("errorText").is_none(),
        "SVG navigation failed: {}",
        navigation["errorText"]
    );
    c.wait("Page.loadEventFired")?;
    std::thread::sleep(Duration::from_secs_f64(seconds));
    let frame = c.call(
        "Page.captureScreenshot",
        json!({"format":"png","captureBeyondViewport":false,"fromSurface":true}),
    )?;
    let png = STANDARD
        .decode(frame["data"].as_str().context("missing captured frame")?)
        .context("decoding captured PNG")?;
    let image = image::load_from_memory_with_format(&png, ImageFormat::Png)
        .context("reading captured frame")?
        .to_rgba8();
    // Graceful close lets Chrome reap renderer children before profile removal.
    let _ = c.call("Browser.close", json!({}));
    let deadline = Instant::now() + Duration::from_secs(2);
    while browser.child.try_wait()?.is_none() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    Ok(image)
}
