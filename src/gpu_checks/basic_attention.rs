use super::*;

pub(super) fn run() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let dim = 8;
    let qh = 4;
    let kvh = 2;
    let tokens = 3;
    let capacity = 5;
    let q: Vec<f32> = (0..qh * dim)
        .map(|i| ((i * 5 % 19) as f32 - 9.) * 0.1)
        .collect();
    let k: Vec<f32> = (0..tokens * kvh * dim)
        .map(|i| ((i * 7 % 23) as f32 - 11.) * 0.1)
        .collect();
    let v: Vec<f32> = (0..tokens * kvh * dim)
        .map(|i| ((i * 11 % 17) as f32 - 8.) * 0.1)
        .collect();
    let gate: Vec<f32> = (0..qh * dim * 2)
        .map(|i| ((i * 3 % 13) as f32 - 6.) * 0.1)
        .collect();
    let dq = DeviceBuffer::from_host(&stream, &q)?;
    let dk = DeviceBuffer::from_host(&stream, &k)?;
    let dv = DeviceBuffer::from_host(&stream, &v)?;
    let dg = DeviceBuffer::from_host(&stream, &gate)?;
    let scores = DeviceBuffer::<f32>::zeroed(&stream, qh * capacity)?;
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, qh * dim)?;
    // SAFETY: all Q/K/V and score extents match their explicit dimensions; each
    // phase has disjoint output ownership and uses the same ordered stream.
    unsafe {
        module.attention.attn_scores(
            &stream,
            LaunchConfig::for_num_elems((qh * tokens * 32) as u32),
            dim,
            qh,
            kvh,
            tokens,
            capacity,
            &dq,
            &dk,
            scores.cu_deviceptr() as *mut f32,
        )?;
        module.attention.softmax(
            &stream,
            LaunchConfig::for_num_elems((qh * 32) as u32),
            qh,
            tokens,
            capacity,
            scores.cu_deviceptr() as *mut f32,
        )?;
        module.attention.attn_values(
            &stream,
            LaunchConfig::for_num_elems((qh * dim) as u32),
            dim,
            qh,
            kvh,
            tokens,
            capacity,
            &scores,
            &dv,
            &dg,
            &mut out,
        )?;
    }
    let mut expected = vec![0f64; qh * dim];
    for h in 0..qh {
        let mut probs: Vec<f64> = (0..tokens)
            .map(|t| {
                (0..dim)
                    .map(|j| {
                        q[h * dim + j] as f64 * k[t * kvh * dim + (h / (qh / kvh)) * dim + j] as f64
                    })
                    .sum::<f64>()
                    / (dim as f64).sqrt()
            })
            .collect();
        let max = probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        probs.iter_mut().for_each(|p| *p = (*p - max).exp());
        let sum: f64 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= sum);
        for j in 0..dim {
            expected[h * dim + j] = (0..tokens)
                .map(|t| probs[t] * v[t * kvh * dim + (h / (qh / kvh)) * dim + j] as f64)
                .sum::<f64>()
                / (1. + (-gate[h * dim * 2 + dim + j] as f64).exp());
        }
    }
    close(&out.to_host_vec(&stream)?, &expected, "grouped attention")?;
    println!("GPU grouped causal attention, softmax and sigmoid gating match f64 reference");
    let parts = tokens.div_ceil(2);
    let cap_parts = 4;
    let partial = DeviceBuffer::<f32>::zeroed(&stream, qh * cap_parts * (dim + 2))?;
    let fused = DeviceBuffer::<f32>::zeroed(&stream, qh * dim)?;
    // SAFETY: two-token context partitions cover the same validated attention
    // problem; merge outputs own one disjoint vector per query head.
    unsafe {
        module.flash.flash_partial::<8, 2>(
            &stream,
            LaunchConfig::for_num_elems((qh * parts * 32) as u32),
            qh,
            kvh,
            tokens,
            parts,
            cap_parts,
            &dq,
            &dk,
            &dv,
            partial.cu_deviceptr() as *mut f32,
        )?;
        module.attention.flash_merge::<8>(
            &stream,
            LaunchConfig::for_num_elems((qh * 32) as u32),
            qh,
            parts,
            cap_parts,
            &partial,
            &dg,
            fused.cu_deviceptr() as *mut f32,
        )?;
    }
    close(
        &fused.to_host_vec(&stream)?,
        &expected,
        "partitioned online attention",
    )?;
    println!("GPU partitioned online attention matches the independent f64 reference");

    Ok(())
}
