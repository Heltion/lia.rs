use super::*;

pub fn cached_rotary() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let dim = 256;
    let heads = 4;
    let batch = 2;
    let rot = 64;
    let position = 8191;
    let base = 10_000_000f32;
    for stride in [256, 512] {
        let input: Vec<f32> = (0..batch * heads * stride)
            .map(|i| ((i * 13 % 37) as f32 - 18.) * 0.07)
            .collect();
        let weight: Vec<f32> = (0..dim).map(|i| 0.5 + (i % 11) as f32 * 0.01).collect();
        let dx = DeviceBuffer::from_host(&stream, &input)?;
        let dw = DeviceBuffer::from_host(&stream, &weight)?;
        let table = DeviceBuffer::<f32>::zeroed(&stream, batch * rot)?;
        let reference = DeviceBuffer::<f32>::zeroed(&stream, batch * heads * dim)?;
        let actual = DeviceBuffer::<f32>::zeroed(&stream, batch * heads * dim)?;
        // SAFETY: both paths normalize the same complete heads. The table has
        // one rotation per token and covers every rotated coordinate.
        unsafe {
            module.rotary.rope_table(
                &stream,
                LaunchConfig::for_num_elems((batch * rot / 2) as u32),
                rot,
                batch,
                position,
                base,
                table.cu_deviceptr() as *mut f32,
            )?;
            module.rotary.norm_rope(
                &stream,
                LaunchConfig::for_num_elems((batch * heads * 32) as u32),
                dim,
                heads,
                stride,
                rot,
                position,
                1e-6,
                base,
                &dx,
                &dw,
                reference.cu_deviceptr() as *mut f32,
            )?;
            module.rotary.norm_rope_cached(
                &stream,
                LaunchConfig::for_num_elems((batch * heads * 32) as u32),
                dim,
                heads,
                stride,
                rot,
                1e-6,
                &dx,
                &dw,
                &table,
                actual.cu_deviceptr() as *mut f32,
            )?;
        }
        let expected: Vec<f64> = reference
            .to_host_vec(&stream)?
            .iter()
            .map(|v| *v as f64)
            .collect();
        close(
            &actual.to_host_vec(&stream)?,
            &expected,
            "cached rotary embedding",
        )?;
    }
    println!(
        "GPU cached rotary embeddings match direct evaluation for Q and K layouts at large positions"
    );
    Ok(())
}
