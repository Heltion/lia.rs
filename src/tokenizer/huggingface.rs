use super::*;
use anyhow::{Context, Result, ensure};
use serde_json::Value as Json;
use std::{fs::File, path::Path};

impl Tokenizer {
    pub fn from_huggingface(directory: impl AsRef<Path>) -> Result<Self> {
        let path = directory.as_ref().join("tokenizer.json");
        let json: Json = serde_json::from_reader(
            File::open(&path).with_context(|| format!("opening {}", path.display()))?,
        )?;
        ensure!(
            json["model"]["type"] == "BPE" && json["decoder"]["type"] == "ByteLevel",
            "unsupported Hugging Face tokenizer model or decoder"
        );
        for name in ["continuing_subword_prefix", "end_of_word_suffix"] {
            ensure!(
                json["model"][name].is_null() || json["model"][name] == "",
                "unsupported tokenizer {name}"
            );
        }
        for name in ["fuse_unk", "byte_fallback", "ignore_merges"] {
            ensure!(json["model"][name] == false, "unsupported tokenizer {name}");
        }
        ensure!(
            (json["normalizer"].is_null() || json["normalizer"]["type"] == "NFC")
                && json["model"]["dropout"].is_null()
                && (json["model"]["unk_token"].is_null() || json["model"]["unk_token"].is_string()),
            "unsupported BPE normalization or fallback"
        );
        let pre = &json["pre_tokenizer"]["pretokenizers"];
        let pre = pre.as_array().context("missing pre-tokenizer sequence")?;
        ensure!(
            json["pre_tokenizer"]["type"] == "Sequence" && pre.len() >= 2,
            "unsupported BPE pre-tokenizer sequence"
        );
        let byte = pre.last().context("missing byte-level pre-tokenizer")?;
        ensure!(
            byte["type"] == "ByteLevel"
                && byte["add_prefix_space"] == false
                && byte["use_regex"] == false,
            "unsupported byte-level pre-tokenizer"
        );
        let mut patterns = Vec::new();
        for split in &pre[..pre.len() - 1] {
            ensure!(
                split["type"] == "Split"
                    && split["behavior"] == "Isolated"
                    && split["invert"] == false,
                "unsupported BPE split behavior"
            );
            patterns.push(
                fancy_regex::Regex::new(
                    split["pattern"]["Regex"]
                        .as_str()
                        .context("missing split expression")?,
                )
                .context("compiling checkpoint split expression")?,
            );
        }
        let vocab = json["model"]["vocab"]
            .as_object()
            .context("missing BPE vocabulary")?;
        let added = json["added_tokens"]
            .as_array()
            .context("missing added tokens")?;
        let mut tokens = Vec::<Value>::new();
        let mut types = Vec::new();
        let mut occupied = Vec::new();
        let mut insert = |id: usize, content: &str, kind: i64| -> Result<()> {
            ensure!(id < 1_000_000, "token ID exceeds supported vocabulary size");
            while tokens.len() <= id {
                tokens.push(Value::String(String::new()));
                types.push(Value::Signed(1));
                occupied.push(false);
            }
            ensure!(
                !occupied[id]
                    || matches!(&tokens[id], Value::String(existing) if existing == content),
                "conflicting tokenizer ID {id}"
            );
            occupied[id] = true;
            tokens[id] = Value::String(content.to_owned());
            types[id] = Value::Signed(kind);
            Ok(())
        };
        for (text, id) in vocab {
            insert(
                usize::try_from(id.as_u64().context("invalid BPE token ID")?)?,
                text,
                1,
            )?;
        }
        let mut always = Vec::new();
        for token in added {
            for name in ["single_word", "lstrip", "rstrip"] {
                ensure!(
                    token[name] == false,
                    "unsupported added-token behavior {name}"
                );
            }
            ensure!(
                token["normalized"] == false || json["normalizer"].is_null(),
                "unsupported normalized added token"
            );
            let id = u32::try_from(token["id"].as_u64().context("missing added-token ID")?)?;
            let text = token["content"]
                .as_str()
                .context("missing added-token content")?;
            // ASCII delimiters cannot compose with adjacent Unicode marks,
            // so normalizing around extraction preserves these added tokens.
            ensure!(
                json["normalizer"].is_null()
                    || (text.is_ascii() && text.starts_with('<') && text.ends_with('>')),
                "unsupported added-token normalization boundary"
            );
            ensure!(!text.is_empty(), "empty added token");
            let special = token["special"]
                .as_bool()
                .context("missing added-token special flag")?;
            insert(id as usize, text, if special { 3 } else { 4 })?;
            if !special {
                always.push((text.to_owned(), id));
            }
        }
        ensure!(occupied.iter().all(|v| *v), "non-contiguous tokenizer IDs");
        let merges = json["model"]["merges"]
            .as_array()
            .context("missing BPE merges")?
            .iter()
            .map(merge)
            .collect::<Result<Vec<_>>>()?;
        let mut tokenizer = Self::from_arrays(&tokens, &types, &merges)?;
        tokenizer.split = patterns;
        always.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));
        tokenizer.added = always;
        tokenizer.normalize_nfc = json["normalizer"]["type"] == "NFC";
        Ok(tokenizer)
    }
}

// Tokenizers serializes the same merge pair as either a string or two strings.
fn merge(value: &Json) -> Result<Value> {
    if let Some(text) = value.as_str() {
        return Ok(Value::String(text.into()));
    }
    let pair = value
        .as_array()
        .context("BPE merge must be a string or pair")?;
    ensure!(pair.len() == 2, "BPE merge must contain exactly two tokens");
    let left = pair[0].as_str().context("invalid left BPE merge token")?;
    let right = pair[1].as_str().context("invalid right BPE merge token")?;
    ensure!(
        !left.is_empty() && !right.is_empty() && !left.contains(' ') && !right.contains(' '),
        "invalid BPE merge token boundary"
    );
    Ok(Value::String(format!("{left} {right}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn merge_representations_have_identical_boundaries() -> Result<()> {
        for input in [serde_json::json!("ĠĠ Ġ"), serde_json::json!(["ĠĠ", "Ġ"])] {
            let Value::String(text) = merge(&input)? else {
                unreachable!()
            };
            assert_eq!(text, "ĠĠ Ġ");
        }
        for invalid in [
            serde_json::json!(["a"]),
            serde_json::json!(["a", 3]),
            serde_json::json!(["a b", "c"]),
            serde_json::json!(["", "a"]),
        ] {
            assert!(merge(&invalid).is_err());
        }
        Ok(())
    }
}
