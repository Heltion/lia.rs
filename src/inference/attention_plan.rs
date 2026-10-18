//! Shared attention geometry for execution, workspace reservation, and prefill chunking.
const HEADS: usize = 24;
const PART_BYTES: usize = 258 * 4;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Plan {
    pub tensor: bool,
    pub span: usize,
    pub parts: usize,
    pub queries: usize,
}

pub(super) fn plan(batch: usize, tokens: usize, mma_decode: bool, tiled: bool) -> Plan {
    debug_assert!(batch > 0 && tokens >= batch);
    let tensor = batch >= if mma_decode { 1 } else { 16 } && (batch >= 16 || tokens - batch >= 512);
    let span = if tensor {
        256
    } else if batch <= 8 {
        64
    } else {
        128
    };
    Plan {
        tensor,
        span,
        parts: tokens.div_ceil(span),
        queries: if tiled && tensor && batch > 8 {
            batch.min(16)
        } else {
            batch
        },
    }
}

pub(super) fn reserved_bytes(
    batch: usize,
    capacity: usize,
    mma_decode: bool,
    tiled: bool,
) -> usize {
    if !tiled {
        return (batch * 384 * 1024)
            .max(8 * HEADS * capacity.div_ceil(if mma_decode { 256 } else { 64 }) * PART_BYTES)
            .max(8 * HEADS * capacity.min(511).div_ceil(64) * PART_BYTES);
    }
    // Include the largest scalar extent just before the small-batch tensor transition.
    (1..=batch)
        .flat_map(|b| [b, capacity, capacity.min(b + 511)].map(move |tokens| (b, tokens)))
        .map(|(b, tokens)| {
            let p = plan(b, tokens, mma_decode, true);
            p.queries * HEADS * p.parts * PART_BYTES
        })
        .max()
        .unwrap_or(0)
}

pub(super) fn chunk(
    remaining: usize,
    maximum: usize,
    position: usize,
    workspace: usize,
    mma_decode: bool,
    tiled: bool,
) -> Option<usize> {
    let mut count = remaining.min(maximum);
    while count > 0 {
        let tokens = position + count;
        let p = plan(count, tokens, mma_decode, tiled);
        let parts = if tiled {
            p.parts
        } else {
            let tile = if count <= 8 {
                64
            } else if count < 16 {
                128
            } else {
                256
            };
            tokens.div_ceil(tile)
        };
        let required = p.queries * HEADS * parts * PART_BYTES;
        if required <= workspace && (tiled || count * tokens <= 8 * 1024 * 1024) {
            return Some(count);
        }
        count = if count > 16 {
            (count - 1) / 16 * 16
        } else {
            count - 1
        };
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserved_workspace_covers_all_valid_batches_and_decode_modes() {
        for capacity in [64, 512, 32768] {
            let maximum = capacity.min(4096);
            for mma in [false, true] {
                let reserved = reserved_bytes(maximum, capacity, mma, true);
                for batch in 1..=maximum {
                    for tokens in [batch, (capacity + batch) / 2, capacity] {
                        let p = plan(batch, tokens, mma, true);
                        assert!(p.span.is_multiple_of(32));
                        assert!(p.queries * HEADS * p.parts * PART_BYTES <= reserved);
                        assert!(p.parts <= plan(batch, tokens, mma, false).parts);
                        if batch <= 8 {
                            assert_eq!(p, plan(batch, tokens, mma, false));
                        }
                        assert_eq!(p.span, plan(batch, tokens, mma, false).span);
                        assert_eq!(p.parts, plan(batch, tokens, mma, false).parts);
                        assert_eq!(
                            chunk(batch, maximum, tokens - batch, reserved, mma, true),
                            Some(batch)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn chunk_planning_preserves_progress_and_legacy_limits() {
        let bytes = 768 * 1024 * 1024;
        for position in [0, 2048, 4096, 8192, 16000, 32000] {
            for remaining in [1, 8, 143, 1668, 7472] {
                let count = chunk(remaining, 2048, position, bytes, true, false).unwrap();
                let tile = if count <= 8 {
                    64
                } else if count < 16 {
                    128
                } else {
                    256
                };
                assert!(count <= remaining && count <= 2048);
                assert!(count * HEADS * (position + count).div_ceil(tile) * PART_BYTES <= bytes);
                assert!(count * (position + count) <= 8 * 1024 * 1024);
            }
        }
        assert_eq!(chunk(1668, 2048, 0, bytes, true, false), Some(1668));
        assert_eq!(chunk(1, 2048, 0, 0, true, true), None);
        let reserved = reserved_bytes(2048, 32768, true, true);
        assert_eq!(chunk(7472, 2048, 8192, reserved, true, true), Some(2048));
        assert!(reserved < 64 * 1024 * 1024);
    }
}
