//! Response channel and tool-call parsing for thinking and ordinary modes.
use super::*;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ordinary_answers_and_json_tool_calls_keep_their_channel() -> Result<()> {
        let answer = parse_with_thinking("42", &json!([]), false)?;
        assert_eq!(answer["content"], "42");
        assert_eq!(answer["reasoning_content"], "");
        let call = parse_with_thinking(
            "<tool_call>{\"name\":\"f\",\"arguments\":{\"x\":3}}</tool_call>",
            &json!([]),
            false,
        )?;
        assert_eq!(call["tool_calls"][0]["function"]["name"], "f");
        assert_eq!(call["tool_calls"][0]["function"]["arguments"], "{\"x\":3}");
        Ok(())
    }

    #[test]
    fn thinking_mode_is_explicit_and_default_is_preserved() -> Result<()> {
        assert!(thinking_enabled(&json!({})));
        assert!(!thinking_enabled(
            &json!({"chat_template_kwargs":{"enable_thinking":false}})
        ));
        let rendered = render_template(
            "{% if enable_thinking %}enabled{% else %}disabled{% endif %}",
            &json!({"messages":[{"role":"user","content":"hi"}],
            "chat_template_kwargs":{"enable_thinking":false}}),
        )?;
        assert_eq!(rendered, "disabled");
        assert!(
            render_template(
                "x",
                &json!({"messages":[{"role":"user","content":"hi"}],
            "chat_template_kwargs":{"enable_thinking":"false"}})
            )
            .is_err()
        );
        Ok(())
    }
}

pub fn parse(raw: &str, tools: &Json) -> Result<Json> {
    parse_with_thinking(raw, tools, true)
}

pub fn parse_with_thinking(raw: &str, tools: &Json, thinking: bool) -> Result<Json> {
    let end = raw.find("</think>");
    let call = raw.find("<tool_call>");
    let (thought, text) = if !thinking {
        ("", raw.trim())
    } else if let Some(call) = call.filter(|&call| end.is_none_or(|end| call < end)) {
        // The tool delimiter also ends reasoning when no explicit end preceded it.
        (&raw[..call], raw[call..].trim())
    } else if let Some(end) = end {
        (&raw[..end], raw[end + "</think>".len()..].trim())
    } else {
        // Every supported chat template opens a thinking span. An unfinished
        // span remains reasoning, including when generation stops naturally.
        (raw, "")
    };
    let mut thought = thought.trim();
    while let Some(rest) = thought.strip_prefix("<think>") {
        thought = rest.trim_start();
    }
    let reasoning = thought.to_string();
    let mut calls = Vec::new();
    let mut remainder = text;
    let mut first_call = None;
    while let Some(start) = remainder.find("<tool_call>") {
        first_call.get_or_insert(text.len() - remainder.len() + start);
        let body_start = start + "<tool_call>".len();
        let end = remainder[body_start..]
            .find("</tool_call>")
            .ok_or_else(|| anyhow!("unclosed tool call"))?
            + body_start;
        let body = remainder[body_start..end].trim();
        if body.starts_with('{') {
            let value: Json = serde_json::from_str(body)?;
            let name = value["name"]
                .as_str()
                .ok_or_else(|| anyhow!("missing JSON tool name"))?;
            let arguments = value.get("arguments").cloned().unwrap_or(json!({}));
            let arguments = if let Some(text) = arguments.as_str() {
                text.to_owned()
            } else {
                serde_json::to_string(&arguments)?
            };
            calls.push(
                json!({"id":format!("lia_call_{}",calls.len()),"type":"function",
                "function":{"name":name,"arguments":arguments}}),
            );
            remainder = &remainder[end + "</tool_call>".len()..];
            continue;
        }
        let func = body
            .strip_prefix("<function=")
            .ok_or_else(|| anyhow!("missing function tag"))?;
        let name_end = func
            .find('>')
            .ok_or_else(|| anyhow!("unclosed function tag"))?;
        let name = &func[..name_end];
        let definition = tools
            .as_array()
            .ok_or_else(|| anyhow!("tools must be an array"))?
            .iter()
            .find_map(|t| {
                let f = t.get("function")?;
                (f.get("name")?.as_str()? == name).then_some(f)
            });
        let parameters = definition
            .and_then(|f| f.get("parameters"))
            .and_then(|p| p.get("properties"))
            .and_then(Json::as_object);
        let mut args = serde_json::Map::new();
        let mut params = &func[name_end + 1..];
        while let Some(start) = params.find("<parameter=") {
            let key_start = start + "<parameter=".len();
            let key_end = params[key_start..]
                .find('>')
                .ok_or_else(|| anyhow!("unclosed parameter tag"))?
                + key_start;
            let key = &params[key_start..key_end];
            let value_end = params[key_end + 1..]
                .find("</parameter>")
                .ok_or_else(|| anyhow!("unclosed parameter value"))?
                + key_end
                + 1;
            let text = params[key_end + 1..value_end].trim();
            let kind = parameters
                .and_then(|p| p.get(key))
                .and_then(|p| p.get("type"))
                .and_then(Json::as_str)
                .unwrap_or("string");
            let value = match kind {
                "string" => Json::String(text.to_owned()),
                "boolean" if text == "True" || text == "False" => Json::Bool(text == "True"),
                _ if text == "None" => Json::Null,
                _ => serde_json::from_str(text).unwrap_or_else(|_| Json::String(text.to_owned())),
            };
            if args.insert(key.to_owned(), value).is_some() {
                return Err(anyhow!("duplicate tool parameter"));
            }
            params = &params[value_end + "</parameter>".len()..];
        }
        // Schema mistakes are generated model output, not malformed API requests.
        // Preserve them for the caller's tool execution and error feedback.
        calls.push(json!({"id":format!("lia_call_{}",calls.len()),"type":"function","function":{"name":name,"arguments":serde_json::to_string(&args)?}}));
        remainder = &remainder[end + "</tool_call>".len()..];
    }
    let content = text[..first_call.unwrap_or(text.len())].trim();
    let mut message = json!({"role":"assistant","content":content,"reasoning_content":reasoning});
    if !calls.is_empty() {
        message["tool_calls"] = Json::Array(calls);
    }
    Ok(message)
}
