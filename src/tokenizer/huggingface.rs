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
            ensure!(json["model"][name] == "", "unsupported tokenizer {name}");
        }
        for name in ["fuse_unk", "byte_fallback", "ignore_merges"] {
            ensure!(json["model"][name] == false, "unsupported tokenizer {name}");
        }
        ensure!(
            (json["normalizer"].is_null() || json["normalizer"]["type"] == "NFC")
                && json["model"]["dropout"].is_null()
                && json["model"]["unk_token"].is_null(),
            "unsupported BPE normalization or fallback"
        );
        let pre = &json["pre_tokenizer"]["pretokenizers"];
        ensure!(
            json["pre_tokenizer"]["type"] == "Sequence"
                && pre.as_array().is_some_and(|a| a.len() == 2)
                && pre[0]["type"] == "Split"
                && pre[0]["behavior"] == "Isolated"
                && pre[0]["invert"] == false
                && pre[1]["type"] == "ByteLevel"
                && pre[1]["add_prefix_space"] == false
                && pre[1]["use_regex"] == false,
            "unsupported BPE pre-tokenizer"
        );
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
            ensure!(!occupied[id], "duplicate tokenizer ID {id}");
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
            for name in ["single_word", "lstrip", "rstrip", "normalized"] {
                ensure!(
                    token[name] == false,
                    "unsupported added-token behavior {name}"
                );
            }
            let id = u32::try_from(token["id"].as_u64().context("missing added-token ID")?)?;
            let text = token["content"]
                .as_str()
                .context("missing added-token content")?;
            // ASCII delimiters cannot compose with adjacent Unicode marks,
            // so normalizing around extraction preserves these added tokens.
            ensure!(
                text.is_ascii() && text.starts_with('<') && text.ends_with('>'),
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
            .map(|v| {
                v.as_str()
                    .map(|s| Value::String(s.to_owned()))
                    .context("unsupported BPE merge representation")
            })
            .collect::<Result<Vec<_>>>()?;
        let mut tokenizer = Self::from_arrays(&tokens, &types, &merges)?;
        ensure!(
            pre[0]["pattern"]["Regex"] == tokenizer.split.as_str(),
            "unsupported BPE splitting expression"
        );
        always.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));
        tokenizer.added = always;
        tokenizer.normalize_nfc = json["normalizer"]["type"] == "NFC";
        Ok(tokenizer)
    }
}
