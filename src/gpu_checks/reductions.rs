use super::*;

pub(super) fn run() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let (cols, rows, batch) = (544usize, 7usize, 4usize);
    let parts = cols.div_ceil(256);
    let bits: Vec<u16> = (0..rows * cols)
        .map(|i| {
            (((i % 29) as f32 - 14.0) * 0.03125)
                .to_bits()
                .wrapping_shr(16) as u16
        })
        .collect();
    let bytes: Vec<u8> = bits.iter().flat_map(|x| x.to_le_bytes()).collect();
    let x: Vec<f32> = (0..batch * cols)
        .map(|i| ((i % 37) as f32 - 18.0) * 0.0125)
        .collect();
    let dw = DeviceBuffer::from_host(&stream, &bytes)?;
    let dx = DeviceBuffer::from_host(&stream, &x)?;
    let partial = DeviceBuffer::<f32>::zeroed(&stream, parts * rows * batch)?;
    let out = DeviceBuffer::<f32>::zeroed(&stream, rows * batch)?;
    // SAFETY: complete BF16 matrices, activation rows and partition outputs.
    unsafe {
        module.matrix.bf16_partial(
            &stream,
            LaunchConfig::for_num_elems((parts * rows * batch * 32) as u32),
            cols,
            rows,
            batch,
            parts,
            &dw,
            &dx,
            partial.cu_deviceptr() as *mut f32,
        )?;
        module.matrix.bf16_merge(
            &stream,
            LaunchConfig::for_num_elems((rows * batch * 32) as u32),
            parts,
            &partial,
            out.cu_deviceptr() as *mut f32,
        )?;
    }
    let mut expected = Vec::new();
    for t in 0..batch {
        for row in 0..rows {
            expected.push(
                (0..cols)
                    .map(|col| {
                        f32::from_bits((bits[row * cols + col] as u32) << 16) as f64
                            * x[t * cols + col] as f64
                    })
                    .sum(),
            );
        }
    }
    close(&out.to_host_vec(&stream)?, &expected, "split BF16 matrix")?;
    println!("GPU split BF16 reduction matches f64 across batches and a partial final partition");
    let vocab: usize = 248320;
    let parts = vocab.div_ceil(1024);
    let mut logits = vec![-0.0f32; 2 * vocab];
    logits[1023] = 3.0;
    logits[1024] = 3.0;
    logits[vocab - 1] = 3.0;
    logits[vocab + 1023] = 0.0;
    logits[vocab + 1024] = 0.0;
    let dl = DeviceBuffer::from_host(&stream, &logits)?;
    let dv = DeviceBuffer::<f32>::zeroed(&stream, 2 * parts)?;
    let di = DeviceBuffer::<u32>::zeroed(&stream, 2 * parts)?;
    // SAFETY: complete vocabulary rows and a disjoint output per partition.
    unsafe {
        module.basic.argmax_partial(
            &stream,
            LaunchConfig::for_num_elems((2 * parts * 32) as u32),
            vocab,
            parts,
            &dl,
            dv.cu_deviceptr() as *mut f32,
            di.cu_deviceptr() as *mut u32,
        )?;
    }
    let values = dv.to_host_vec(&stream)?;
    let ids = di.to_host_vec(&stream)?;
    for row in 0..2 {
        let actual = values[row * parts..(row + 1) * parts]
            .iter()
            .zip(&ids[row * parts..(row + 1) * parts])
            .max_by(|a, b| a.0.total_cmp(b.0).then(a.1.cmp(b.1)))
            .unwrap()
            .1;
        let expected = logits[row * vocab..(row + 1) * vocab]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        if *actual != expected {
            return Err(anyhow!(
                "GPU greedy reduction disagrees on ties or signed zero"
            ));
        }
    }
    println!(
        "GPU vocabulary reduction matches total float order, partition boundaries and last-index ties"
    );
    Ok(())
}
