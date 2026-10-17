use super::*;

pub fn check() -> anyhow::Result<()> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let n = 1003;
    let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.125 - 63.0).collect();
    let y: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * -0.25).collect();
    let dx = DeviceBuffer::from_host(&stream, &x)?;
    let dy = DeviceBuffer::from_host(&stream, &y)?;
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, n)?;
    let module = kernels::load(&context)?;
    // SAFETY: the one-dimensional grid covers the output; the kernel guards its
    // tail and every input has exactly the output length.
    unsafe {
        module.basic.residual(
            &stream,
            LaunchConfig::for_num_elems(n as u32),
            &dx,
            &dy,
            &mut out,
        )?;
    }
    let actual = out.to_host_vec(&stream)?;
    for (i, actual) in actual.iter().enumerate() {
        if *actual != x[i] + y[i] {
            return Err(anyhow!("GPU residual mismatch at {i}"));
        }
    }
    println!("GPU residual: {n} elements match the Rust CPU reference exactly");
    let formats = [
        (10, 84),
        (12, 144),
        (14, 210),
        (16, 66),
        (17, 74),
        (18, 98),
        (19, 50),
        (21, 110),
        (22, 82),
        (23, 136),
        (29, 56),
    ];
    for (kind, block_bytes) in formats {
        let mut input: Vec<u8> = (0..block_bytes * 3)
            .map(|i| ((i * 73 + 19) % 256) as u8)
            .collect();
        for block in input.chunks_exact_mut(block_bytes) {
            let offset = if kind == 10 {
                80
            } else if kind == 14 {
                208
            } else {
                0
            };
            if kind != 29 {
                block[offset..offset + 2].copy_from_slice(&0x3800u16.to_le_bytes());
                if kind == 10 || kind == 12 {
                    block[offset + 2..offset + 4].copy_from_slice(&0x3400u16.to_le_bytes());
                }
            } else {
                for (j, nibble) in [0u16, 0, 8, 3].iter().enumerate() {
                    let pos = 48 + j * 2;
                    let bits = u16::from_le_bytes([block[pos], block[pos + 1]]);
                    block[pos..pos + 2]
                        .copy_from_slice(&((bits & 4095) | (nibble << 12)).to_le_bytes());
                }
            }
        }
        let mut expected = Vec::new();
        for block in input.chunks_exact(block_bytes) {
            expected.extend(crate::quant::decode_block(kind, block)?);
        }
        let din = DeviceBuffer::from_host(&stream, &input)?;
        let mut dout = DeviceBuffer::<f32>::zeroed(&stream, expected.len())?;
        // SAFETY: input has three complete blocks, output has three groups of 256;
        // the one-dimensional grid matches the guarded output index.
        unsafe {
            module.basic.decode(
                &stream,
                LaunchConfig::for_num_elems(expected.len() as u32),
                kind,
                block_bytes,
                &din,
                &mut dout,
            )?;
        }
        let actual = dout.to_host_vec(&stream)?;
        let mut max_error = 0.0f32;
        for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
            let error = (a - e).abs();
            if !a.is_finite() || !e.is_finite() || error > 1e-5 * e.abs().max(1.0) {
                return Err(anyhow!(
                    "GPU storage type {kind} mismatch at {i}: {a} vs {e}"
                ));
            }
            max_error = max_error.max(error);
        }
        println!(
            "GPU storage type {kind}: {} values, maximum error {max_error}",
            expected.len()
        );
        let x: Vec<f32> = (0..256).map(|j| (j as f32 - 117.0) * 0.001).collect();
        let dx = DeviceBuffer::from_host(&stream, &x)?;
        let dout = DeviceBuffer::<f32>::zeroed(&stream, 3)?;
        // SAFETY: one complete warp per row, all inputs cover three 256-element
        // rows, and each row is written once by its warp's lane zero.
        unsafe {
            module.matrix.matvec(
                &stream,
                LaunchConfig::for_num_elems(96),
                kind,
                block_bytes,
                256,
                256,
                3,
                &din,
                &dx,
                dout.cu_deviceptr() as *mut f32,
            )?;
        }
        let actual = dout.to_host_vec(&stream)?;
        for (row, a) in actual.iter().enumerate() {
            let expected: f64 = expected[row * 256..(row + 1) * 256]
                .iter()
                .zip(&x)
                .map(|(w, x)| *w as f64 * *x as f64)
                .sum();
            if (*a as f64 - expected).abs() > 2e-4 * expected.abs().max(1.0) {
                return Err(anyhow!(
                    "GPU matvec type {kind}, row {row}: {a} vs {expected}"
                ));
            }
        }
        println!("GPU matvec type {kind}: three rows match f64 reference");
    }
    let dim = 65;
    let groups = 3;
    let input: Vec<f32> = (0..dim * groups)
        .map(|i| (i as f32 - 101.0) * 0.003)
        .collect();
    let weight: Vec<f32> = (0..dim).map(|i| 0.75 + i as f32 * 0.002).collect();
    let dx = DeviceBuffer::from_host(&stream, &input)?;
    let dw = DeviceBuffer::from_host(&stream, &weight)?;
    let output = DeviceBuffer::<f32>::zeroed(&stream, input.len())?;
    // SAFETY: each full warp owns a group; input and output cover three groups,
    // and every lane only writes its own strided elements.
    unsafe {
        module.basic.rms_norm(
            &stream,
            LaunchConfig::for_num_elems((groups * 32) as u32),
            dim,
            groups,
            1e-6,
            &dx,
            &dw,
            output.cu_deviceptr() as *mut f32,
        )?;
    }
    let actual = output.to_host_vec(&stream)?;
    for group in 0..groups {
        let sum: f64 = input[group * dim..(group + 1) * dim]
            .iter()
            .map(|x| (*x as f64).powi(2))
            .sum();
        let scale = 1.0 / (sum / dim as f64 + 1e-6).sqrt();
        for j in 0..dim {
            let expected = input[group * dim + j] as f64 * scale * weight[j] as f64;
            if (actual[group * dim + j] as f64 - expected).abs() > 1e-5 {
                return Err(anyhow!("GPU RMS norm mismatch"));
            }
        }
    }
    println!("GPU RMS norm: three non-aligned groups match f64 reference");
    Ok(())
}
