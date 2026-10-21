//! Validated launches for the sixty-four-query shared-KV attention tile.
use super::*;
use anyhow::{Result, ensure};

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch<const GROUPED: bool>(
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
        query_heads > 0 && kv_heads > 0 && query_heads.is_multiple_of(kv_heads),
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
    let prepared =
        module
            .flash_mma64
            .prepare_flash_mma64::<GROUPED>(cuda_core::simt::LaunchConfig1D::new(
                (queries.div_ceil(64) * heads * parts) as u32,
                1024,
                96256,
            ))?;
    // SAFETY: complete head/tile geometry, live non-overlapping caches and
    // output storage, and the prepared launch's dynamic-shared contract.
    unsafe {
        module.flash_mma64.flash_mma64::<GROUPED>(
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

    #[test]
    fn wide_attention_preserves_causal_tails_partials_and_f64_reference() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        for (batch, tokens, qh, kvh, span, start, count) in [
            (1usize, 33usize, 4usize, 2usize, 32usize, 0usize, 1usize),
            (65, 129, 4, 2, 64, 3, 59),
            (129, 513, 16, 2, 256, 16, 65),
            (33, 2049, 16, 2, 256, 0, 33),
            (257, 4097, 16, 2, 256, 128, 64),
            (17, 1025, 24, 4, 128, 1, 15),
        ] {
            let parts = tokens.div_ceil(span);
            let q = (0..batch * qh * 256)
                .map(|i| ((i * 17 % 37) as f32 - 18.0) * 0.2)
                .collect::<Vec<_>>();
            let k = (0..tokens * kvh * 256)
                .map(|i| (((i * 23 % 43) as f32 - 21.0) * 0.15) as f16)
                .collect::<Vec<_>>();
            let v = (0..tokens * kvh * 256)
                .map(|i| (((i * 31 % 41) as f32 - 20.0) * 0.05) as f16)
                .collect::<Vec<_>>();
            let dq = DeviceBuffer::from_host(&stream, &q)?;
            let dk = DeviceBuffer::from_host(
                &stream,
                &k.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            )?;
            let dv = DeviceBuffer::from_host(
                &stream,
                &v.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            )?;
            let size = count * qh * parts * 258;
            let sentinel = f32::from_bits(0x3f12_3456);
            let blank = vec![sentinel; size + 32];
            let mut baseline = None;
            for grouped in [false, true] {
                for wide in [false, true] {
                    let partial = DeviceBuffer::from_host(&stream, &blank)?;
                    macro_rules! run {
                        ($grouped:literal) => {
                            if wide {
                                launch::<$grouped>(&module, &stream, qh, kvh, tokens, span,
                                    parts, start, count, &dq, &dk, &dv, &partial)?;
                            } else {
                                let groups = if $grouped { qh / kvh } else { 1 };
                                let heads = if $grouped { kvh } else { qh };
                                let prepared = module.flash_mma.prepare_flash_mma::<$grouped, true, 32>(
                                    cuda_core::simt::LaunchConfig1D::new(
                                        ((count * groups).div_ceil(32) * heads * parts) as u32, 512, 60160))?;
                                // SAFETY: validated complete heads/caches and guarded output window.
                                unsafe { module.flash_mma.flash_mma::<$grouped, true, 32>(
                                    &stream, &prepared, qh, kvh, tokens, span, parts, start,
                                    count, &dq, &dk, &dv, partial.cu_deviceptr() as *mut f32)?; }
                            }
                        };
                    }
                    if grouped {
                        run!(true);
                    } else {
                        run!(false);
                    }
                    let values = partial.to_host_vec(&stream)?;
                    assert!(
                        values[size..]
                            .iter()
                            .all(|v| v.to_bits() == sentinel.to_bits()),
                        "partial-window guard overwritten: batch={batch}, count={count}"
                    );
                    if let Some(expected) = &baseline {
                        let expected: &Vec<f32> = expected;
                        for (index, (&a, &b)) in values[..size].iter().zip(expected).enumerate() {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "partial={index}, batch={batch}, tokens={tokens}, grouped={grouped}, wide={wide}"
                            );
                        }
                    } else {
                        baseline = Some(values[..size].to_vec());
                    }
                    // Sample boundary queries, both end heads and separated output
                    // coordinates against an independent f64 causal softmax.
                    for local in [0, count - 1] {
                        let query = start + local;
                        let valid = tokens - batch + query + 1;
                        for head in [0, qh - 1] {
                            let kh = head / (qh / kvh);
                            let scores = (0..valid)
                                .map(|t| {
                                    (0..256)
                                        .map(|j| {
                                            q[(query * qh + head) * 256 + j] as f64
                                                * k[(t * kvh + kh) * 256 + j] as f64
                                        })
                                        .sum::<f64>()
                                        / 16.0
                                })
                                .collect::<Vec<_>>();
                            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                            let probs = scores.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
                            let total = probs.iter().sum::<f64>();
                            let base = (local * qh + head) * parts * 258;
                            let partial_max = (0..parts)
                                .map(|p| values[base + p * 258 + 256] as f64)
                                .fold(f64::NEG_INFINITY, f64::max);
                            let denom = (0..parts)
                                .map(|p| {
                                    values[base + p * 258 + 257] as f64
                                        * (values[base + p * 258 + 256] as f64 - partial_max).exp()
                                })
                                .sum::<f64>();
                            for column in [0, 31, 128, 255] {
                                let reference = (0..valid)
                                    .map(|t| probs[t] * v[(t * kvh + kh) * 256 + column] as f64)
                                    .sum::<f64>()
                                    / total;
                                let actual = (0..parts)
                                    .map(|p| {
                                        values[base + p * 258 + column] as f64
                                            * (values[base + p * 258 + 256] as f64 - partial_max)
                                                .exp()
                                    })
                                    .sum::<f64>()
                                    / denom;
                                assert!(
                                    (actual - reference).abs() < 3e-5 * reference.abs().max(1.0),
                                    "causal softmax {actual} vs {reference}: query={query}, head={head}, column={column}"
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
