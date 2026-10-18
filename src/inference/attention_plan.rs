//! Shared attention geometry for execution, workspace reservation, and prefill chunking.
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
    let span = if tensor && batch <= 8 {
        // Expose enough independent GQA partitions at short contexts, then
        // amortize partial reductions over larger spans as the context grows.
        if tokens <= 2048 {
            32
        } else if tokens <= 8192 {
            64
        } else {
            128
        }
    } else if tensor {
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
    heads: usize,
    mma_decode: bool,
    tiled: bool,
) -> usize {
    // A span increase can reduce the partition count. Include the preceding
    // context boundary as well as the last scalar-only prefix for every batch.
    let maximum_batch = if tiled { batch } else { batch.min(8) };
    let required = (1..=maximum_batch)
        .flat_map(|b| {
            [
                b,
                capacity,
                capacity.min(b + 511),
                capacity.min(2048).max(b),
                capacity.min(8192).max(b),
            ]
            .map(move |tokens| (b, tokens))
        })
        .map(|(b, tokens)| {
            let p = plan(b, tokens, mma_decode, tiled);
            p.queries * heads * p.parts * PART_BYTES
        })
        .max()
        .unwrap_or(0);
    if tiled {
        required
    } else {
        required.max(batch * 384 * 1024)
    }
}

pub(super) fn chunk(
    remaining: usize,
    maximum: usize,
    position: usize,
    workspace: usize,
    heads: usize,
    mma_decode: bool,
    tiled: bool,
) -> Option<usize> {
    let mut count = remaining.min(maximum);
    while count > 0 {
        let tokens = position + count;
        let p = plan(count, tokens, mma_decode, tiled);
        let required = p.queries * heads * p.parts * PART_BYTES;
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
    fn reserved_workspace_covers_batches_heads_and_span_boundaries() {
        for capacity in [64, 512, 2048, 2049, 8192, 8193, 32768, 65536, 262144] {
            let maximum = capacity.min(4096);
            for heads in [16, 24] {
                for mma in [false, true] {
                    let reserved = reserved_bytes(maximum, capacity, heads, mma, true);
                    for batch in 1..=maximum {
                        for tokens in [
                            batch,
                            (capacity + batch) / 2,
                            capacity,
                            capacity.min(2048).max(batch),
                            capacity.min(8192).max(batch),
                        ] {
                            let p = plan(batch, tokens, mma, true);
                            assert!(p.span.is_multiple_of(32));
                            assert!(p.queries * heads * p.parts * PART_BYTES <= reserved);
                            assert_eq!(p.span, plan(batch, tokens, mma, false).span);
                            assert_eq!(p.parts, plan(batch, tokens, mma, false).parts);
                            if batch <= 8 {
                                assert_eq!(p, plan(batch, tokens, mma, false));
                            }
                            assert_eq!(
                                chunk(batch, maximum, tokens - batch, reserved, heads, mma, true),
                                Some(batch)
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn chunk_planning_preserves_progress_and_legacy_limits() {
        let bytes = 768 * 1024 * 1024;
        for position in [0, 2048, 4096, 8192, 16000, 32000, 65536, 262000] {
            for remaining in [1, 8, 143, 1668, 7472] {
                let count = chunk(remaining, 2048, position, bytes, 24, true, false).unwrap();
                let p = plan(count, position + count, true, false);
                assert!(count <= remaining && count <= 2048);
                assert!(p.queries * 24 * p.parts * PART_BYTES <= bytes);
                assert!(count * (position + count) <= 8 * 1024 * 1024);
            }
        }
        assert_eq!(chunk(1668, 2048, 0, bytes, 24, true, false), Some(1668));
        assert_eq!(chunk(1, 2048, 0, 0, 24, true, true), None);
        let reserved = reserved_bytes(2048, 32768, 24, true, true);
        assert_eq!(
            chunk(7472, 2048, 8192, reserved, 24, true, true),
            Some(2048)
        );
        assert!(reserved < 64 * 1024 * 1024);
    }

    #[test]
    fn decode_partition_sizes_increase_only_at_declared_boundaries() {
        for (tokens, span) in [(513, 32), (2048, 32), (2049, 64), (8192, 64), (8193, 128)] {
            assert_eq!(plan(1, tokens, true, true).span, span);
        }
        assert_eq!(plan(1, 512, true, true).span, 64);
        assert_eq!(plan(16, 2048, true, true).span, 256);
    }
}
