//! Deterministic BPE pieces are memoized with a bounded, tokenizer-local working set.
use super::*;
impl Tokenizer {
    pub(super) fn encode_piece(&self, piece: &str, output: &mut Vec<u32>) -> anyhow::Result<()> {
        let cacheable = piece.len() <= 256;
        if cacheable {
            let cache = self
                .bpe_cache
                .lock()
                .map_err(|_| anyhow!("reading BPE cache: poisoned lock"))?;
            if let Some(ids) = cache.get(piece) {
                output.extend_from_slice(ids);
                return Ok(());
            }
        }
        let mut parts: Vec<String> = piece
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
        let ids = parts
            .into_iter()
            .map(|part| {
                self.vocab
                    .get(&part)
                    .copied()
                    .ok_or_else(|| anyhow!("BPE piece absent from vocabulary"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        output.extend_from_slice(&ids);
        if cacheable {
            let mut cache = self
                .bpe_cache
                .lock()
                .map_err(|_| anyhow!("updating BPE cache: poisoned lock"))?;
            if cache.len() >= 4096 {
                cache.clear();
            }
            cache.insert(piece.to_owned(), ids);
        }
        Ok(())
    }
}
