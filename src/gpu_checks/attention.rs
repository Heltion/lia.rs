use super::*;

pub fn long_attention() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let dim = 256;
    let qh = 4;
    let kvh = 2;
    let batch = 2;
    let tokens = 129;
    let parts = 2;
    let cap = 3;
    let q: Vec<f32> = (0..batch * qh * dim)
        .map(|i| ((i * 17 % 37) as f32 - 18.) * 0.2)
        .collect();
    let k: Vec<f32> = (0..tokens * kvh * dim)
        .map(|i| ((i * 23 % 43) as f32 - 21.) * 0.15)
        .collect();
    let v: Vec<f32> = (0..tokens * kvh * dim)
        .map(|i| ((i * 31 % 41) as f32 - 20.) * 0.05)
        .collect();
    let k: Vec<f32> = k.into_iter().map(|x| (x as f16) as f32).collect();
    let v: Vec<f32> = v.into_iter().map(|x| (x as f16) as f32).collect();
    let gate: Vec<f32> = (0..batch * qh * dim * 2)
        .map(|i| ((i * 13 % 29) as f32 - 14.) * 0.2)
        .collect();
    let mut expected = vec![0f64; batch * qh * dim];
    for t in 0..batch {
        let valid = tokens - batch + t + 1;
        for h in 0..qh {
            let g = t * qh + h;
            let kh = h / (qh / kvh);
            let mut p: Vec<f64> = (0..valid)
                .map(|s| {
                    (0..dim)
                        .map(|j| q[g * dim + j] as f64 * k[s * kvh * dim + kh * dim + j] as f64)
                        .sum::<f64>()
                        / 16.
                })
                .collect();
            let max = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            p.iter_mut().for_each(|p| *p = (*p - max).exp());
            let sum: f64 = p.iter().sum();
            for j in 0..dim {
                expected[g * dim + j] = (0..valid)
                    .map(|s| p[s] * v[s * kvh * dim + kh * dim + j] as f64)
                    .sum::<f64>()
                    / sum
                    / (1. + (-gate[g * dim * 2 + dim + j] as f64).exp());
            }
        }
    }
    let dq = DeviceBuffer::from_host(&stream, &q)?;
    let dk = DeviceBuffer::from_host(&stream, &k)?;
    let dv = DeviceBuffer::from_host(&stream, &v)?;
    let dg = DeviceBuffer::from_host(&stream, &gate)?;
    let partial = DeviceBuffer::<f32>::zeroed(&stream, batch * qh * cap * (dim + 2))?;
    let out = DeviceBuffer::<f32>::zeroed(&stream, batch * qh * dim)?;
    // SAFETY: two queries share a 127-token prefix and append two causal tokens;
    // the second partition is empty for query zero and contains one token for
    // query one. All outputs cover their complete head dimensions.
    unsafe {
        module.flash.flash_partial::<256, 128>(
            &stream,
            LaunchConfig::for_num_elems((batch * qh * parts * 32) as u32),
            qh,
            kvh,
            tokens,
            parts,
            cap,
            &dq,
            &dk,
            &dv,
            partial.cu_deviceptr() as *mut f32,
        )?;
        module.attention.flash_merge::<256>(
            &stream,
            LaunchConfig::for_num_elems((batch * qh * 32) as u32),
            batch * qh,
            parts,
            cap,
            &partial,
            &dg,
            out.cu_deviceptr() as *mut f32,
        )?;
    }
    close(
        &out.to_host_vec(&stream)?,
        &expected,
        "256-dimension causal online attention",
    )?;
    println!(
        "GPU 256-dimensional online attention matches f64 across a context partition boundary and two causal queries"
    );
    let k16 = DeviceBuffer::<u16>::zeroed(&stream, k.len())?;
    let v16 = DeviceBuffer::<u16>::zeroed(&stream, v.len())?;
    // SAFETY: exact cache extents, full causal context and one producer per cache element.
    unsafe {
        module.rotary.cache_kv_half(
            &stream,
            LaunchConfig::for_num_elems(k.len() as u32),
            kvh * dim,
            0,
            &dk,
            &dv,
            k16.cu_deviceptr() as *mut u16,
            v16.cu_deviceptr() as *mut u16,
        )?;
        module.flash_half.flash_partial_half::<256, 128>(
            &stream,
            LaunchConfig::for_num_elems((batch * qh * parts * 32) as u32),
            qh,
            kvh,
            tokens,
            parts,
            cap,
            &dq,
            &k16,
            &v16,
            partial.cu_deviceptr() as *mut f32,
        )?;
        module.attention.flash_merge::<256>(
            &stream,
            LaunchConfig::for_num_elems((batch * qh * 32) as u32),
            batch * qh,
            parts,
            cap,
            &partial,
            &dg,
            out.cu_deviceptr() as *mut f32,
        )?;
    }
    close(
        &out.to_host_vec(&stream)?,
        &expected,
        "half-cache partitioned attention",
    )?;
    println!("GPU f16 KV cache and partitioned attention match the independent reference");
    Ok(())
}
