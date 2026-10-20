//! Rendering the model-provided chat template and parsing its tool-call format.
use crate::gguf::{Model, Value};
use anyhow::{Result, anyhow};
use serde_json::{Value as Json, json};

struct SpacedJson;
impl serde_json::ser::Formatter for SpacedJson {
    fn begin_array_value<W: std::io::Write + ?Sized>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_key<W: std::io::Write + ?Sized>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_value<W: std::io::Write + ?Sized>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b": ")
    }
}
pub fn render(model: &Model, request: &Json) -> Result<String> {
    let template = match model.metadata.get("tokenizer.chat_template") {
        Some(Value::String(s)) => s,
        _ => return Err(anyhow!("missing GGUF chat template")),
    };
    render_template(template, request)
}

pub fn render_template(template: &str, request: &Json) -> Result<String> {
    let mut env = minijinja::Environment::new();
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function(
        "raise_exception",
        |message: String| -> std::result::Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                message,
            ))
        },
    );
    env.add_filter(
        "tojson",
        |value: minijinja::Value| -> std::result::Result<minijinja::Value, minijinja::Error> {
            let mut bytes = Vec::new();
            let mut serializer = serde_json::Serializer::with_formatter(&mut bytes, SpacedJson);
            serde::Serialize::serialize(&value, &mut serializer).map_err(|e| {
                minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string())
            })?;
            Ok(minijinja::Value::from_safe_string(
                String::from_utf8(bytes).map_err(|e| {
                    minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string())
                })?,
            ))
        },
    );
    env.add_template("chat", template)?;
    let messages = request
        .get("messages")
        .ok_or_else(|| anyhow!("missing chat messages"))?;
    if !messages.is_array() || messages.as_array().is_none_or(Vec::is_empty) {
        return Err(anyhow!("chat messages must be a nonempty array"));
    }
    let mut messages = messages.clone();
    for message in messages
        .as_array_mut()
        .ok_or_else(|| anyhow!("invalid messages"))?
    {
        if message["role"] == "assistant"
            && let Some(reasoning) = message.get("reasoning").filter(|v| v.is_string())
        {
            message["reasoning_content"] = reasoning.clone();
        }
        if let Some(calls) = message.get_mut("tool_calls").and_then(Json::as_array_mut) {
            for call in calls {
                if let Some(arguments) = call
                    .get_mut("function")
                    .and_then(|f| f.get_mut("arguments"))
                    && let Some(text) = arguments.as_str()
                {
                    *arguments = serde_json::from_str(text)?;
                }
            }
        }
    }
    // The OpenAI tool envelope exposes these three function fields to chat
    // templates. BFCL response schemas remain in the external tool definitions.
    let tools:Vec<Json>=request.get("tools").and_then(Json::as_array).into_iter().flatten().map(|tool| {
        let f=&tool["function"];
        json!({"type":"function","function":{"name":f["name"],"description":f.get("description").cloned().unwrap_or(json!("")),"parameters":f["parameters"]}})
    }).collect();
    for message in messages.as_array_mut().unwrap() {
        if message["role"] == "developer" {
            message["role"] = json!("system");
        }
    }
    let mut context = json!({"messages": messages,"tools":tools,"add_generation_prompt":true,"enable_thinking":true,"reasoning_effort":request.get("reasoning_effort").cloned().unwrap_or(json!("xhigh")),"preserve_reasoning":true});
    if let Some(kwargs) = request.get("chat_template_kwargs") {
        for (k, v) in kwargs
            .as_object()
            .ok_or_else(|| anyhow!("chat_template_kwargs must be an object"))?
        {
            context[k] = v.clone();
        }
    }
    if context["enable_thinking"] != true {
        return Err(anyhow!(
            "thinking must remain enabled for this implementation"
        ));
    }
    Ok(env.get_template("chat")?.render(context)?)
}

#[cfg(any(feature = "cuda", test))]
pub fn parse(raw: &str, tools: &Json) -> Result<Json> {
    let (reasoning, text) = match raw.split_once("</think>") {
        Some((a, b)) => (a.trim_start_matches("<think>").trim().to_string(), b.trim()),
        None => (String::new(), raw.trim()),
    };
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn assistant_reasoning_fields_render_identically() -> Result<()> {
        let template = "{% for m in messages %}{{ m.reasoning_content|default('') }}:{{ m.content }}{% endfor %}";
        let old = json!({"messages":[{"role":"assistant","content":"answer","reasoning_content":"thought"}]});
        let modern =
            json!({"messages":[{"role":"assistant","content":"answer","reasoning":"thought"}]});
        assert_eq!(render_template(template, &old)?, "thought:answer");
        assert_eq!(
            render_template(template, &modern)?,
            render_template(template, &old)?
        );
        let both = json!({"messages":[{"role":"assistant","content":"answer","reasoning":"current","reasoning_content":"old"}]});
        assert_eq!(render_template(template, &both)?, "current:answer");
        let null = json!({"messages":[{"role":"assistant","content":"answer","reasoning":null,"reasoning_content":"thought"}]});
        assert_eq!(
            render_template(template, &null)?,
            render_template(template, &old)?
        );
        Ok(())
    }
    #[test]
    fn tool_values_and_thinking() {
        let tools = json!([{"type":"function","function":{"name":"f","parameters":{"properties":{"s":{"type":"string"},"n":{"type":"integer"}},"required":["s","n"]}}}]);
        let raw = "Check this carefully.</think>\n<tool_call>\n<function=f>\n<parameter=s>line one\nline two</parameter>\n<parameter=n>3</parameter>\n</function>\n</tool_call>";
        let parsed = parse(raw, &tools).unwrap();
        assert_eq!(parsed["reasoning_content"], "Check this carefully.");
        let args: Json = serde_json::from_str(
            parsed["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(args, json!({"s":"line one\nline two","n":3}));
        assert!(parse("<tool_call><function=f>", &tools).is_err());
    }
    #[test]
    fn python_boolean_and_null_parameters() {
        let tools = json!([{"type":"function","function":{"name":"f","parameters":{"properties":{"enabled":{"type":"boolean"},"value":{"type":"integer"}}}}}]);
        let raw = "</think><tool_call><function=f><parameter=enabled>False</parameter><parameter=value>None</parameter></function></tool_call>";
        let parsed = parse(raw, &tools).unwrap();
        let args: Json = serde_json::from_str(
            parsed["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(args, json!({"enabled":false,"value":null}));
    }
    #[test]
    fn generated_schema_errors_remain_tool_calls() -> Result<()> {
        let tools = json!([{"type":"function","function":{"name":"cd","parameters":{
            "properties":{"folder":{"type":"string"},"count":{"type":"integer"}},
            "required":["folder"]}}}]);
        let raw = "</think><tool_call><function=cd><parameter=path>..</parameter><parameter=count>not a number</parameter></function></tool_call>";
        let parsed = parse(raw, &tools)?;
        let call = &parsed["tool_calls"][0]["function"];
        let args: Json = serde_json::from_str(call["arguments"].as_str().unwrap())?;
        assert_eq!(call["name"], "cd");
        assert_eq!(args, json!({"path":"..","count":"not a number"}));
        let unknown = parse(
            "<tool_call><function=unknown><parameter=x>raw value</parameter></function></tool_call>",
            &tools,
        )?;
        assert_eq!(unknown["tool_calls"][0]["function"]["name"], "unknown");
        Ok(())
    }
}
