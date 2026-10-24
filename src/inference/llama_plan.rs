//! Query groups preserve request boundaries while controlling attention workspace traffic.
pub(super) struct Group {
    pub start: usize,
    pub count: usize,
    pub maximum: usize,
}
pub(super) struct Tile {
    pub start: usize,
    pub rows: usize,
    pub parts: usize,
    pub span: usize,
    pub tensor: bool,
    pub compact: bool,
}

pub(super) fn tiles(
    groups: &[Group],
    batch: usize,
    maximum: usize,
    flash: bool,
    compact: bool,
    full_span: bool,
) -> Vec<Tile> {
    if flash
        && compact
        && groups.iter().all(|g| g.count <= 8)
        && (batch > 8 || groups.iter().all(|g| g.count == 1))
    {
        let span = super::llama::decode_span(maximum);
        let parts = super::llama::decode_parts(maximum, span);
        return (0..batch)
            .step_by(64)
            .map(|start| Tile {
                start,
                rows: (batch - start).min(64),
                parts,
                span,
                tensor: true,
                compact: true,
            })
            .collect();
    }
    if batch <= 8 || !flash || groups.iter().all(|g| g.count == 1) {
        let span = if batch <= 8 || groups.iter().all(|g| g.count == 1) {
            super::llama::decode_span(maximum)
        } else {
            256
        };
        let parts = super::llama::decode_parts(maximum, span);
        return (0..batch)
            .step_by(64)
            .map(|start| Tile {
                start,
                rows: (batch - start).min(64),
                parts,
                span,
                tensor: false,
                compact: false,
            })
            .collect();
    }
    // Each tensor tile owns eight token rows. Aligned request boundaries let
    // one launch cover the cohort while deriving the KV sequence per tile.
    // A single partition writes directly, so the larger row extent needs no
    // additional partial-attention storage.
    if maximum <= 4096 && groups.iter().all(|g| g.start.is_multiple_of(8)) {
        return vec![Tile {
            start: 0,
            rows: batch,
            parts: 1,
            span: 4096,
            tensor: true,
            compact: false,
        }];
    }
    let mut result = Vec::new();
    for group in groups {
        let tensor = group.count >= 2;
        let chunk = if tensor { 1024 } else { 64 };
        for start in (group.start..group.start + group.count).step_by(chunk) {
            let rows = (group.start + group.count - start).min(chunk);
            let maximum = group.maximum - (group.start + group.count - start - rows);
            let span = if tensor && full_span && rows >= 512 {
                maximum.div_ceil(64) * 64
            } else if tensor {
                4096
            } else {
                super::llama::decode_span(maximum)
            };
            result.push(Tile {
                start,
                rows,
                parts: maximum.div_ceil(span),
                span,
                tensor,
                compact: false,
            });
        }
    }
    result
}
