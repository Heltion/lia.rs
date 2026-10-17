//! Byte-level BPE, with Qwen 3.5 Unicode pre-tokenization.
use crate::gguf::{Model, Value};
use anyhow::anyhow;
use std::collections::HashMap;

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    pieces: Vec<Vec<u8>>,
    ranks: HashMap<(String, String), usize>,
    encode_byte: [char; 256],
    split: fancy_regex::Regex,
    special: Vec<(String, u32)>,
}
impl Tokenizer {
    pub fn from_model(model: &Model) -> anyhow::Result<Self> {
        if !matches!(model.metadata.get("tokenizer.ggml.pre"), Some(Value::String(s)) if s == "qwen35")
        {
            return Err(anyhow!("unsupported tokenizer pre-tokenization"));
        }
        let array = |key: &str| -> anyhow::Result<&Vec<Value>> {
            match model.metadata.get(key) {
                Some(Value::Array { values, .. }) => Ok(values),
                _ => Err(anyhow!("missing array {key}")),
            }
        };
        let mut encode_byte = ['\0'; 256];
        let mut extra = 256;
        for b in 0..=255u32 {
            let visible =
                (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
            let code = if visible {
                b
            } else {
                let c = extra;
                extra += 1;
                c
            };
            encode_byte[b as usize] =
                char::from_u32(code).ok_or_else(|| anyhow!("invalid byte map"))?;
        }
        let inverse: HashMap<char, u8> = encode_byte
            .iter()
            .enumerate()
            .map(|(b, c)| (*c, b as u8))
            .collect();
        let tokens = array("tokenizer.ggml.tokens")?;
        let types = array("tokenizer.ggml.token_type")?;
        if tokens.len() != types.len() {
            return Err(anyhow!("token types length mismatch"));
        }
        let mut vocab = HashMap::new();
        let mut pieces = Vec::new();
        let mut special = Vec::new();
        for (id, (token, kind)) in tokens.iter().zip(types).enumerate() {
            let Value::String(token) = token else {
                return Err(anyhow!("non-string token"));
            };
            let Value::Signed(kind) = kind else {
                return Err(anyhow!("invalid token type"));
            };
            if *kind == 3 || *kind == 4 {
                special.push((token.clone(), id as u32));
            }
            let piece = if *kind == 3 || *kind == 4 {
                token.as_bytes().to_vec()
            } else {
                token
                    .chars()
                    .map(|c| {
                        inverse
                            .get(&c)
                            .copied()
                            .ok_or_else(|| anyhow!("invalid byte token"))
                    })
                    .collect::<Result<Vec<_>, _>>()?
            };
            vocab.insert(token.clone(), id as u32);
            pieces.push(piece);
        }
        special.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));
        let mut ranks = HashMap::new();
        for (rank, merge) in array("tokenizer.ggml.merges")?.iter().enumerate() {
            let Value::String(merge) = merge else {
                return Err(anyhow!("non-string merge"));
            };
            let (a, b) = merge
                .split_once(' ')
                .ok_or_else(|| anyhow!("invalid BPE merge"))?;
            ranks.insert((a.to_owned(), b.to_owned()), rank);
        }
        let split = fancy_regex::Regex::new(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        )?;
        Ok(Self {
            vocab,
            pieces,
            ranks,
            encode_byte,
            split,
            special,
        })
    }
    pub fn encode(&self, text: &str, parse_special: bool) -> anyhow::Result<Vec<u32>> {
        let mut out = Vec::new();
        let mut offset = 0;
        while offset < text.len() {
            let next = if parse_special {
                self.special
                    .iter()
                    .filter_map(|(s, id)| {
                        text[offset..].find(s).map(|i| (offset + i, s.len(), *id))
                    })
                    .min_by_key(|(i, len, _)| (*i, std::cmp::Reverse(*len)))
            } else {
                None
            };
            let end = next.map_or(text.len(), |(i, _, _)| i);
            for found in self.split.find_iter(&text[offset..end]) {
                let found = found?;
                let mut parts: Vec<String> = found
                    .as_str()
                    .bytes()
                    .map(|b| self.encode_byte[b as usize].to_string())
                    .collect();
                loop {
                    let best = parts
                        .windows(2)
                        .enumerate()
                        .filter_map(|(i, pair)| {
                            self.ranks
                                .get(&(pair[0].clone(), pair[1].clone()))
                                .map(|rank| (*rank, i))
                        })
                        .min();
                    let Some((_, index)) = best else { break };
                    let right = parts.remove(index + 1);
                    parts[index].push_str(&right);
                }
                for part in parts {
                    out.push(
                        *self
                            .vocab
                            .get(&part)
                            .ok_or_else(|| anyhow!("BPE piece absent from vocabulary"))?,
                    );
                }
            }
            if let Some((i, len, id)) = next {
                out.push(id);
                offset = i + len;
            } else {
                break;
            }
        }
        Ok(out)
    }
    pub fn decode(&self, ids: &[u32]) -> anyhow::Result<Vec<u8>> {
        let mut out = Vec::new();
        for id in ids {
            out.extend_from_slice(
                self.pieces
                    .get(*id as usize)
                    .ok_or_else(|| anyhow!("token ID out of range"))?,
            );
        }
        Ok(out)
    }
}
