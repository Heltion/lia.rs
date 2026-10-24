//! Checkpoint-defined byte-level BPE and text pre-tokenization.
use crate::gguf::{Model, Value};
use anyhow::anyhow;
use std::{
    collections::{HashMap, HashSet},
    sync::Mutex,
};
use unicode_normalization::UnicodeNormalization;
mod bpe;
mod huggingface;

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    pieces: Vec<Vec<u8>>,
    ranks: HashMap<(String, String), usize>,
    encode_byte: [char; 256],
    split: Vec<fancy_regex::Regex>,
    special: Vec<(String, u32)>,
    added: Vec<(String, u32)>,
    normalize_nfc: bool,
    control_ids: HashSet<u32>,
    bpe_cache: Mutex<HashMap<String, Vec<u32>>>,
}
impl Tokenizer {
    pub fn is_control(&self, token: u32) -> bool {
        self.control_ids.contains(&token)
    }
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
        Self::from_arrays(
            array("tokenizer.ggml.tokens")?,
            array("tokenizer.ggml.token_type")?,
            array("tokenizer.ggml.merges")?,
        )
    }

    fn from_arrays(tokens: &[Value], types: &[Value], merges: &[Value]) -> anyhow::Result<Self> {
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
        if tokens.len() != types.len() {
            return Err(anyhow!("token types length mismatch"));
        }
        let mut vocab = HashMap::new();
        let mut pieces = Vec::new();
        let mut special = Vec::new();
        let mut control_ids = HashSet::new();
        for (id, (token, kind)) in tokens.iter().zip(types).enumerate() {
            let Value::String(token) = token else {
                return Err(anyhow!("non-string token"));
            };
            let Value::Signed(kind) = kind else {
                return Err(anyhow!("invalid token type"));
            };
            if *kind == 3 {
                control_ids.insert(id as u32);
            }
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
        for (rank, merge) in merges.iter().enumerate() {
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
            split: vec![split],
            special,
            added: Vec::new(),
            normalize_nfc: false,
            control_ids,
            bpe_cache: Mutex::new(HashMap::new()),
        })
    }
    pub fn encode(&self, text: &str, parse_special: bool) -> anyhow::Result<Vec<u32>> {
        let normalized = self.normalize_nfc.then(|| text.nfc().collect::<String>());
        let text = normalized.as_deref().unwrap_or(text);
        let mut out = Vec::new();
        let mut offset = 0;
        while offset < text.len() {
            let next = {
                let recognized = if parse_special {
                    &self.special
                } else {
                    &self.added
                };
                recognized
                    .iter()
                    .filter_map(|(s, id)| {
                        text[offset..].find(s).map(|i| (offset + i, s.len(), *id))
                    })
                    .min_by_key(|(i, len, _)| (*i, std::cmp::Reverse(*len)))
            };
            let end = next.map_or(text.len(), |(i, _, _)| i);
            let mut pieces = vec![&text[offset..end]];
            for pattern in &self.split {
                let mut next = Vec::new();
                for piece in pieces {
                    let mut consumed = 0;
                    for found in pattern.find_iter(piece) {
                        let found = found?;
                        if found.start() > consumed {
                            next.push(&piece[consumed..found.start()]);
                        }
                        if !found.as_str().is_empty() {
                            next.push(found.as_str());
                        }
                        consumed = found.end();
                    }
                    if consumed < piece.len() {
                        next.push(&piece[consumed..]);
                    }
                }
                pieces = next;
            }
            for piece in pieces {
                self.encode_piece(piece, &mut out)?;
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
    /// Decode user-visible text while retaining added tokens not marked special.
    pub fn decode_text(&self, ids: &[u32]) -> anyhow::Result<Vec<u8>> {
        let kept: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|id| !self.control_ids.contains(id))
            .collect();
        self.decode(&kept)
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn text_decoding_skips_control_tokens_without_removing_structural_tags() -> anyhow::Result<()> {
        let tokens =
            ["a", "<|tts_eos|>", "<think>", "<tool_call>"].map(|s| Value::String(s.into()));
        let types = [1, 3, 4, 4].map(Value::Signed);
        let tokenizer = Tokenizer::from_arrays(&tokens, &types, &[])?;
        assert_eq!(
            tokenizer.decode(&[0, 1, 2, 3])?,
            b"a<|tts_eos|><think><tool_call>"
        );
        assert_eq!(
            tokenizer.decode_text(&[0, 1, 2, 3])?,
            b"a<think><tool_call>"
        );
        assert!(tokenizer.decode_text(&[999]).is_err());
        Ok(())
    }
}
