//! Experimental BF16 attention launch contract; not selected by inference.
use super::*;
use anyhow::{Result, ensure};

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch<const GROUPED: bool, const ROWS: u32>(
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<CudaStream>,
    query_heads: usize,
    kv_heads: usize,
    tokens: usize,
    span: usize,
    parts: usize,
    query_start: usize,
    query_count: usize,
    q: &DeviceBuffer<f32>,
    k: &DeviceBuffer<u16>,
    v: &DeviceBuffer<u16>,
    partial: &DeviceBuffer<f32>,
) -> Result<()> {
    ensure!(
        matches!(ROWS, 16 | 64)
            && query_heads > 0
            && kv_heads > 0
            && query_heads.is_multiple_of(kv_heads),
        "invalid wide attention heads"
    );
    ensure!(
        !q.is_empty() && q.len().is_multiple_of(query_heads * 256),
        "invalid wide attention query extent"
    );
    let batch = q.len() / (query_heads * 256);
    ensure!(
        tokens >= batch
            && span > 0
            && span.is_multiple_of(32)
            && parts == tokens.div_ceil(span)
            && query_start < batch
            && query_count > 0
            && query_count <= batch - query_start
            && k.len() >= tokens * kv_heads * 256
            && v.len() >= tokens * kv_heads * 256
            && partial.len() >= query_count * query_heads * parts * 258,
        "invalid wide attention cache, window or partial extent"
    );
    let (heads, queries) = if GROUPED {
        (kv_heads, query_count * (query_heads / kv_heads))
    } else {
        (query_heads, query_count)
    };
    let prepared = module.flash_bf16.prepare_flash_bf16::<GROUPED, ROWS>(
        cuda_core::simt::LaunchConfig1D::new(
            (queries.div_ceil(ROWS as usize) * heads * parts) as u32,
            ROWS * 16,
            ROWS * 712 + 16896,
        ),
    )?;
    // SAFETY: complete head/tile geometry, live non-overlapping caches and
    // output storage, and the prepared launch's dynamic-shared contract.
    unsafe {
        module.flash_bf16.flash_bf16::<GROUPED, ROWS>(
            stream,
            &prepared,
            query_heads,
            kv_heads,
            tokens,
            span,
            parts,
            query_start,
            query_count,
            q,
            k,
            v,
            partial.cu_deviceptr() as *mut f32,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round(x: f32) -> f32 {
        let bits = x.to_bits();
        f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
    }

    #[test]
    fn bf16_attention_matches_independent_online_oracle_and_guards() -> Result<()> {
        assert_eq!(round(1.00390625), 1.0);
        assert_eq!(round(1.01171875), 1.015625);
        assert_eq!(round(-1.00390625), -1.0);
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        for (batch, tokens, qh, kvh, span, start, count) in [
            (1usize, 33usize, 4usize, 2usize, 32usize, 0usize, 1usize),
            (65, 129, 4, 2, 64, 3, 59),
            (129, 513, 16, 2, 256, 16, 65),
            (17, 1025, 24, 4, 128, 1, 15),
            (257, 4097, 16, 2, 256, 128, 64),
            (129, 4097, 16, 2, 4128, 1, 127),
        ] {
            let parts = tokens.div_ceil(span);
            // BF16-rounded non-dyadic Q and dyadic K keep these small QK
            // products and sums exactly representable;
            // the oracle independently models online softmax and BF16 P.
            let q = (0..batch * qh * 256)
                .map(|i| ((i * 17 % 37) as f32 - 18.0) / 31.0)
                .collect::<Vec<_>>();
            let k = (0..tokens * kvh * 256)
                .map(|i| round(((i * 23 % 43) as f32 - 21.0) / 64.0))
                .collect::<Vec<_>>();
            let v = (0..tokens * kvh * 256)
                .map(|i| round(((i * 31 % 41) as f32 - 20.0) / 32.0))
                .collect::<Vec<_>>();
            let dq = DeviceBuffer::from_host(&stream, &q)?;
            let dk = DeviceBuffer::from_host(
                &stream,
                &k.iter()
                    .map(|v| (v.to_bits() >> 16) as u16)
                    .collect::<Vec<_>>(),
            )?;
            let dv = DeviceBuffer::from_host(
                &stream,
                &v.iter()
                    .map(|v| (v.to_bits() >> 16) as u16)
                    .collect::<Vec<_>>(),
            )?;
            let size = count * qh * parts * 258;
            let sentinel = f32::from_bits(0x3f12_3456);
            let mut baseline = None;
            for rows in [16, 64] {
                for grouped in [false, true] {
                    let partial = DeviceBuffer::from_host(&stream, &vec![sentinel; size + 32])?;
                    macro_rules! run {
                        ($g:literal,$rows:literal) => {
                            launch::<$g, $rows>(
                                &module, &stream, qh, kvh, tokens, span, parts, start, count, &dq,
                                &dk, &dv, &partial,
                            )?
                        };
                    }
                    match (rows, grouped) {
                        (16, false) => run!(false, 16),
                        (16, true) => run!(true, 16),
                        (64, false) => run!(false, 64),
                        (64, true) => run!(true, 64),
                        _ => unreachable!(),
                    }
                    let values = partial.to_host_vec(&stream)?;
                    assert!(
                        values[size..]
                            .iter()
                            .all(|v| v.to_bits() == sentinel.to_bits())
                    );
                    let bits = values[..size]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>();
                    if let Some(reference) = &baseline {
                        assert_eq!(&bits, reference, "GQA BF16 bits");
                    } else {
                        baseline = Some(bits);
                    }
                    for local in [0, count - 1] {
                        let query = start + local;
                        let valid = tokens - batch + query + 1;
                        for head in [0, qh - 1] {
                            let kh = head / (qh / kvh);
                            let scores = (0..valid)
                                .map(|t| {
                                    (0..256)
                                        .map(|c| {
                                            round(q[(query * qh + head) * 256 + c]) as f64
                                                * k[(t * kvh + kh) * 256 + c] as f64
                                        })
                                        .sum::<f64>()
                                        / 16.0
                                })
                                .collect::<Vec<_>>();
                            for part in 0..parts {
                                let mut maximum = f64::NEG_INFINITY;
                                let mut total = 0.0;
                                let mut numerators = [0.0; 4];
                                let columns = [0, 31, 128, 255];
                                let end = ((part + 1) * span).min(valid);
                                let mut position = part * span;
                                while position < end {
                                    let tile_end = (position + 32).min(end);
                                    let next_max = scores[position..tile_end]
                                        .iter()
                                        .copied()
                                        .fold(maximum, f64::max);
                                    let factor = if maximum.is_finite() {
                                        (maximum - next_max).exp()
                                    } else {
                                        0.0
                                    };
                                    total *= factor;
                                    for n in &mut numerators {
                                        *n *= factor;
                                    }
                                    for t in position..tile_end {
                                        let probability = (scores[t] - next_max).exp();
                                        total += probability;
                                        let packed_probability = round(probability as f32) as f64;
                                        for (n, c) in numerators.iter_mut().zip(columns) {
                                            *n += packed_probability
                                                * v[(t * kvh + kh) * 256 + c] as f64;
                                        }
                                    }
                                    maximum = next_max;
                                    position += 32;
                                }
                                let base = ((local * qh + head) * parts + part) * 258;
                                if maximum.is_finite() {
                                    assert!((values[base + 256] as f64 - maximum).abs() < 1e-6);
                                } else {
                                    assert_eq!(values[base + 256], f32::NEG_INFINITY);
                                }
                                assert!(
                                    (values[base + 257] as f64 - total).abs()
                                        < 2e-5 * total.abs().max(1.0)
                                );
                                for (c, expected) in columns.into_iter().zip(numerators) {
                                    let actual = values[base + c] as f64;
                                    assert!(
                                        (actual - expected).abs() < 2e-4 * expected.abs().max(1.0),
                                        "BF16 numerator {actual} vs {expected}; batch={batch},tokens={tokens},part={part},col={c}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
