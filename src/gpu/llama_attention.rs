//! Validated launches for causal 128-dimensional, sixteen-to-two-head GQA.
use super::*;
use anyhow::{Result, ensure};
#[derive(Clone, Copy)]
pub(crate) struct Config {
    pub start: usize,
    pub queries: usize,
    pub position: usize,
    pub parts: usize,
    pub span: usize,
    pub paged: bool,
    pub stride: usize,
    pub tensor: bool,
    pub compact: bool,
    pub wide: bool,
    pub direct: bool,
}

/// # Safety
/// Page/sequence IDs and logical positions must name initialized KV entries.
/// Each eight-token prefill tile must belong to one sequence; compact rows have independent metadata.
/// Buffers remain live on the stream.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn partial(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: Config,
    positions: &DeviceBuffer<u32>,
    sequences: &DeviceBuffer<u32>,
    pages: &DeviceBuffer<u32>,
    q: &DeviceBuffer<f32>,
    k: &DeviceBuffer<u16>,
    v: &DeviceBuffer<u16>,
    output: &DeviceBuffer<f32>,
) -> Result<()> {
    ensure!(
        q.len().is_multiple_of(2048)
            && config.queries > 0
            && config.start + config.queries <= q.len() / 2048,
        "invalid GQA query extent"
    );
    ensure!(
        k.len() == v.len() && k.len().is_multiple_of(256),
        "invalid GQA cache extent"
    );
    ensure!(
        config.parts > 0
            && config.parts <= 1024
            && config.span > 0
            && config.span.is_multiple_of(32),
        "invalid GQA partitions"
    );
    ensure!(
        output.len()
            >= if config.direct {
                (config.start + config.queries) * 2048
            } else {
                config.queries * 16 * config.parts * 130
            },
        "GQA partial workspace exhausted"
    );
    ensure!(
        q.cu_deviceptr().is_multiple_of(16)
            && k.cu_deviceptr().is_multiple_of(16)
            && v.cu_deviceptr().is_multiple_of(16)
            && output.cu_deviceptr().is_multiple_of(8),
        "unaligned GQA buffers"
    );
    if config.paged {
        ensure!(
            positions.len() >= q.len() / 2048
                && sequences.len() >= q.len() / 2048
                && config.stride > 0
                && pages.len() >= config.stride,
            "invalid GQA metadata extents"
        );
    } else {
        ensure!(
            config.position + config.start + config.queries <= k.len() / 256,
            "GQA context exceeds cache"
        );
    }
    ensure!(
        !config.direct || (config.tensor && config.parts == 1),
        "direct attention needs one tensor partition"
    );
    ensure!(
        !config.wide || (!config.compact && config.span.is_multiple_of(64)),
        "wide attention partition extent"
    );
    unsafe {
        if config.tensor {
            ensure!(
                !config.compact || config.queries <= 64,
                "compact attention query extent"
            );
            macro_rules! launch {
                ($kernel:ident, $threads:literal, $rows:literal) => {
                    module.llama_flash.$kernel(
                        stream,
                        LaunchConfig {
                            grid_dim: (
                                (config.queries * 8).div_ceil($rows) as u32
                                    * 2
                                    * config.parts as u32,
                                1,
                                1,
                            ),
                            block_dim: ($threads, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        config.start,
                        config.queries,
                        config.position,
                        config.parts,
                        config.span,
                        config.paged,
                        config.stride,
                        positions,
                        sequences,
                        pages,
                        q,
                        k,
                        v,
                        config.direct,
                        output.cu_deviceptr() as *mut f32,
                    )?
                };
            }
            if config.compact {
                launch!(llama_flash_decode, 32, 8);
            } else if config.wide {
                launch!(llama_flash_wide, 128, 64);
            } else {
                launch!(llama_flash128, 128, 64);
            }
        } else {
            module.llama.llama_attention_partial(
                stream,
                LaunchConfig::for_num_elems((config.queries * 16 * config.parts * 32) as u32),
                config.start,
                config.queries,
                config.position,
                config.parts,
                config.span,
                config.paged,
                config.stride,
                positions,
                sequences,
                pages,
                q,
                k,
                v,
                output.cu_deviceptr() as *mut f32,
            )?;
        }
    }
    Ok(())
}

/// # Safety
/// Partials must come from the matching successful launch, and buffers remain live.
pub(crate) unsafe fn merge(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: Config,
    partial: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
) -> Result<()> {
    ensure!(
        partial.len() >= config.queries * 16 * config.parts * 130
            && output.len() >= (config.start + config.queries) * 2048,
        "invalid GQA merge extents"
    );
    unsafe {
        if config.parts <= 16 {
            module.llama_merge.llama_merge_small(
                stream,
                LaunchConfig {
                    grid_dim: ((config.queries * 16).div_ceil(4) as u32, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                },
                config.start,
                config.queries,
                config.parts,
                partial,
                output.cu_deviceptr() as *mut f32,
            )?;
        } else {
            module.llama.llama_attention_merge(
                stream,
                LaunchConfig {
                    grid_dim: ((config.queries * 16) as u32, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                },
                config.start,
                config.queries,
                config.parts,
                partial,
                output.cu_deviceptr() as *mut f32,
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn value(sequence: usize, token: usize, head: usize, column: usize, plane: usize) -> f32 {
        ((sequence * 7 + token * 17 + head * 23 + column * 13 + plane * 11) % 257) as f32 / 256.0
            - 0.5
    }
    #[test]
    #[ignore = "requires a CUDA device"]
    fn scalar_and_tensor_paged_attention_match_independent_f64_reference() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        let capacity = 16416;
        let stride = capacity / 32;
        let positions: Vec<u32> = (4090..4098).chain(16377..16385).chain([8191]).collect();
        let sequences: Vec<u32> = (0..8)
            .map(|_| 0)
            .chain((0..8).map(|_| 1))
            .chain([0])
            .collect();
        let count = positions.len();
        let tables: Vec<_> = (0..2)
            .flat_map(|sequence| {
                (0..stride).map(move |p| ((1 - sequence) * stride + stride - 1 - p) as u32)
            })
            .collect();
        let mut keys = vec![0_u16; capacity * 2 * 256];
        let mut values = keys.clone();
        for sequence in 0..2 {
            for token in 0..capacity {
                let physical = tables[sequence * stride + token / 32] as usize * 32 + token % 32;
                for head in 0..2 {
                    for column in 0..128 {
                        let at = (physical * 2 + head) * 128 + column;
                        keys[at] = (value(sequence, token, head, column, 0) as f16).to_bits();
                        values[at] = (value(sequence, token, head, column, 1) as f16).to_bits();
                    }
                }
            }
        }
        let queries: Vec<_> = (0..count * 2048)
            .map(|i| (i * 19 % 61) as f32 / 64.0 - 30.0 / 64.0)
            .collect();
        let mut expected = vec![0_f64; queries.len()];
        for row in 0..count {
            for head in 0..16 {
                let count = positions[row] as usize + 1;
                let scores: Vec<_> = (0..count)
                    .map(|token| {
                        (0..128)
                            .map(|column| {
                                f64::from(queries[(row * 16 + head) * 128 + column])
                                    * f64::from(value(
                                        sequences[row] as usize,
                                        token,
                                        head / 8,
                                        column,
                                        0,
                                    ))
                            })
                            .sum::<f64>()
                            / 128_f64.sqrt()
                    })
                    .collect();
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let total: f64 = scores.iter().map(|s| (s - maximum).exp()).sum();
                for (token, score) in scores.into_iter().enumerate() {
                    let weight = (score - maximum).exp() / total;
                    for column in 0..128 {
                        expected[(row * 16 + head) * 128 + column] += weight
                            * f64::from(value(sequences[row] as usize, token, head / 8, column, 1));
                    }
                }
            }
        }
        let q = DeviceBuffer::from_host(&stream, &queries)?;
        let k = DeviceBuffer::from_host(&stream, &keys)?;
        let v = DeviceBuffer::from_host(&stream, &values)?;
        let pos = DeviceBuffer::from_host(&stream, &positions)?;
        let seq = DeviceBuffer::from_host(&stream, &sequences)?;
        let pages = DeviceBuffer::from_host(&stream, &tables)?;
        let work = DeviceBuffer::zeroed(&stream, count * 16 * 144 * 130)?;
        let output = DeviceBuffer::zeroed(&stream, queries.len())?;
        for (tensor, compact, wide, direct, grouped) in [
            (false, false, false, false, false),
            (true, false, false, false, false),
            (true, false, true, false, false),
            (true, true, false, false, false),
            (true, false, true, true, false),
            (true, false, false, false, true),
            (true, false, true, true, true),
        ] {
            if tensor && !compact && !grouped {
                for row in 0..count {
                    let config = Config {
                        start: row,
                        queries: 1,
                        position: 0,
                        parts: if direct {
                            1
                        } else {
                            (positions[row] as usize + 1).div_ceil(4096)
                        },
                        span: if direct { 32768 } else { 4096 },
                        paged: true,
                        stride,
                        tensor,
                        compact,
                        wide,
                        direct,
                    };
                    unsafe {
                        partial(
                            &module,
                            &stream,
                            config,
                            &pos,
                            &seq,
                            &pages,
                            &q,
                            &k,
                            &v,
                            if direct { &output } else { &work },
                        )?;
                        if !direct {
                            merge(&module, &stream, config, &work, &output)?;
                        }
                    }
                }
            } else {
                let config = Config {
                    start: 0,
                    queries: count,
                    position: 0,
                    parts: if direct {
                        1
                    } else if grouped {
                        5
                    } else {
                        144
                    },
                    span: if direct {
                        32768
                    } else if grouped {
                        4096
                    } else {
                        128
                    },
                    paged: true,
                    stride,
                    tensor,
                    compact,
                    wide,
                    direct,
                };
                unsafe {
                    partial(
                        &module,
                        &stream,
                        config,
                        &pos,
                        &seq,
                        &pages,
                        &q,
                        &k,
                        &v,
                        if direct { &output } else { &work },
                    )?;
                    if !direct {
                        merge(&module, &stream, config, &work, &output)?;
                    }
                }
            }
            let actual = output.to_host_vec(&stream)?;
            let mut maximum = 0_f64;
            for (&a, &b) in actual.iter().zip(&expected) {
                ensure!(a.is_finite(), "nonfinite GQA value");
                maximum = maximum.max((f64::from(a) - b).abs());
            }
            println!(
                "attention tensor={tensor} compact={compact} wide={wide} direct={direct} grouped={grouped} max_abs={maximum}"
            );
            ensure!(
                maximum < 3e-5,
                "attention differs from independent reference"
            );
        }
        Ok(())
    }
}
