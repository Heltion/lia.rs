//! MCP stdio transport. Image results are delivered as image content, not SVG text.
use super::*;
use std::io::{BufRead, Write};

pub fn serve() -> Result<()> {
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    loop {
        let mut line = String::new();
        let bytes = input
            .by_ref()
            .take(1024 * 1024 + 1)
            .read_line(&mut line)
            .context("reading MCP request")?;
        if bytes == 0 {
            return Ok(());
        }
        anyhow::ensure!(bytes <= 1024 * 1024, "MCP request exceeds 1 MiB limit");
        let request: Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(_) => {
                writeln!(output, "{}", error(Value::Null, -32700, "Invalid JSON"))?;
                output.flush()?;
                continue;
            }
        };
        if let Some(response) = dispatch(&request) {
            serde_json::to_writer(&mut output, &response).context("encoding MCP response")?;
            writeln!(output)?;
            output.flush().context("flushing MCP response")?;
        }
    }
}

fn error(id: Value, code: i32, message: &str) -> Value {
    json!({"jsonrpc":"2.0","id":id,"error":{"code":code,"message":message}})
}

fn dispatch(request: &Value) -> Option<Value> {
    let id = request.get("id")?.clone();
    let result = match request["method"].as_str().unwrap_or("") {
        "initialize" => {
            let requested = request["params"]["protocolVersion"].as_str().unwrap_or("");
            let version = match requested {
                "2024-11-05" | "2025-03-26" | "2025-06-18" => requested,
                _ => "2025-06-18",
            };
            json!({"protocolVersion":version,"capabilities":{"tools":{}},"serverInfo":{"name":"lia-image-tools","version":env!("CARGO_PKG_VERSION")}})
        }
        "ping" => json!({}),
        "tools/list" => json!({"tools":[{
            "name":"read_image",
            "description":"View a local SVG, PNG, or JPEG as an image. Animated SVG defaults to its first frame; optionally set time_seconds to sample a later frame with an isolated headless browser (960x720). No desktop capture. Static SVG is rasterized in Rust. Scripts and external resources are unsupported. Choose whether and when to inspect; this tool does not assess quality.",
            "inputSchema":{"type":"object","properties":{"path":{"type":"string","description":"Local image file path; an absolute path is recommended."},"time_seconds":{"type":"number","minimum":0,"maximum":120,"description":"Animation time in seconds after SVG load; omitted means the first frame."}},"required":["path"],"additionalProperties":false},
            "annotations":{"readOnlyHint":true,"destructiveHint":false,"openWorldHint":false}
        }]}),
        "tools/call" => {
            let params = &request["params"];
            if params["name"] != "read_image" {
                return Some(error(id, -32602, "Unknown tool"));
            }
            let result = params["arguments"]["path"]
                .as_str()
                .context("read_image requires a string path")
                .and_then(|path| {
                    let time = params["arguments"]
                        .get("time_seconds")
                        .map(|v| v.as_f64().context("time_seconds must be numeric"))
                        .transpose()?;
                    read_at(path, time)
                });
            match result {
                Ok(picture) => picture.tool_result(),
                Err(error) => {
                    json!({"isError":true,"content":[{"type":"text","text":format!("{error:#}")}]})
                }
            }
        }
        _ => return Some(error(id, -32601, "Method not found")),
    };
    Some(json!({"jsonrpc":"2.0","id":id,"result":result}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mcp_negotiates_version_and_exposes_image_tool() {
        let reply = dispatch(
            &json!({"id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26"}}),
        )
        .unwrap();
        assert_eq!(reply["result"]["protocolVersion"], "2025-03-26");
        let tools = dispatch(&json!({"id":2,"method":"tools/list"})).unwrap();
        assert_eq!(tools["result"]["tools"][0]["name"], "read_image");
        assert!(dispatch(&json!({"method":"notifications/initialized"})).is_none());
        let bad = dispatch(
            &json!({"id":3,"method":"tools/call","params":{"name":"read_image","arguments":{}}}),
        )
        .unwrap();
        assert_eq!(bad["result"]["isError"], true);
    }
}
