use super::*;

pub fn integer_matmul() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let cols = 2304;
    let rows = 19;
    for batch in [1, 2, 4, 8] {
        let x: Vec<f32> = (0..batch * cols)
            .map(|i| {
                if i < 32 {
                    0.0
                } else {
                    ((i * 71 % 257) as f32 - 128.0) * 0.003
                }
            })
            .collect();
        let mut quant = Vec::new();
        let mut scales = Vec::new();
        for group in x.as_chunks::<32>().0 {
            let max = group.iter().map(|v| v.abs()).fold(0f32, f32::max);
            let scale = if max == 0.0 { 1.0 } else { max / 127.0 };
            scales.push(scale);
            quant.extend(group.iter().map(|v| (v / scale).round_ties_even() as i8));
        }
        let dx = DeviceBuffer::from_host(&stream, &x)?;
        let packed = DeviceBuffer::<u32>::zeroed(&stream, x.len() / 4)?;
        let ds = DeviceBuffer::<f32>::zeroed(&stream, x.len() / 32)?;
        // SAFETY: complete 32-element groups, four packed bytes per word and one
        // scaling value per group. Every output has a unique producer.
        unsafe {
            module.integer.quantize_activation(
                &stream,
                LaunchConfig::for_num_elems(x.len() as u32),
                &dx,
                packed.cu_deviceptr() as *mut u32,
                ds.cu_deviceptr() as *mut f32,
            )?;
        }
        let actual = packed.to_host_vec(&stream)?;
        let expected: Vec<u32> = quant
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| u32::from_le_bytes([v[0] as u8, v[1] as u8, v[2] as u8, v[3] as u8]))
            .collect();
        if actual != expected || ds.to_host_vec(&stream)? != scales {
            return Err(anyhow!("GPU activation quantization mismatch"));
        }
        println!(
            "GPU activation quantization matches nearest-even CPU reference, including all-zero groups"
        );
        for (kind, size) in [
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
        ] {
            let mut weights: Vec<u8> = (0..rows * (cols / 256) * size)
                .map(|i| ((i * 73 + 19) % 256) as u8)
                .collect();
            for b in weights.chunks_exact_mut(size) {
                let offset = if kind == 10 {
                    80
                } else if kind == 14 {
                    208
                } else {
                    0
                };
                if kind == 29 {
                    for (j, nib) in [0u16, 0, 8, 3].iter().enumerate() {
                        let p = 48 + j * 2;
                        let v = u16::from_le_bytes([b[p], b[p + 1]]);
                        b[p..p + 2].copy_from_slice(&((v & 4095) | (nib << 12)).to_le_bytes());
                    }
                } else {
                    b[offset..offset + 2].copy_from_slice(&0x3800u16.to_le_bytes());
                    if kind == 10 || kind == 12 {
                        b[offset + 2..offset + 4].copy_from_slice(&0x3400u16.to_le_bytes());
                    }
                }
            }
            let mut expected = vec![0f64; batch * rows];
            let mut sum_abs = vec![0f64; batch * rows];
            for row in 0..rows {
                let mut decoded = Vec::new();
                for block in weights[row * (cols / 256) * size..(row + 1) * (cols / 256) * size]
                    .chunks_exact(size)
                {
                    decoded.extend(crate::quant::decode_block(kind, block)?);
                }
                for t in 0..batch {
                    sum_abs[t * rows + row] = (0..cols)
                        .map(|j| {
                            (decoded[j] as f64
                                * quant[t * cols + j] as f64
                                * scales[(t * cols + j) / 32] as f64)
                                .abs()
                        })
                        .sum();
                    expected[t * rows + row] = (0..cols)
                        .map(|j| {
                            decoded[j] as f64
                                * quant[t * cols + j] as f64
                                * scales[(t * cols + j) / 32] as f64
                        })
                        .sum();
                }
            }
            let dw = DeviceBuffer::from_host(&stream, &weights)?;
            let out = DeviceBuffer::<f32>::zeroed(&stream, batch * rows)?;
            for (split, row_group) in [(1, 1), (1, 2), (4, 2)] {
                // SAFETY: complete matrices and row-aligned activation groups;
                // the launch uses one or four complete warps per output row.
                unsafe {
                    crate::gpu::launch_int_matmul(
                        &module,
                        &stream,
                        kind,
                        cols,
                        rows,
                        batch,
                        split,
                        row_group,
                        &dw,
                        &packed,
                        &ds,
                        out.cu_deviceptr() as *mut f32,
                    )?;
                }
                let actual = out.to_host_vec(&stream)?;
                // Integer dot products are exact. The remaining error is
                // bounded by rounded scaling, per-lane accumulation and
                // the warp/CTA reduction tree, including cancellation.
                let operations = cols.div_ceil(512) + 16;
                let nu = operations as f64 * f32::EPSILON as f64;
                let gamma = nu / (1.0 - nu);
                for (i, ((value, reference), magnitude)) in
                    actual.iter().zip(&expected).zip(&sum_abs).enumerate()
                {
                    if !value.is_finite()
                        || (*value as f64 - reference).abs() > gamma * magnitude + 1e-6
                    {
                        return Err(anyhow!(
                            "integer matrix rounding bound failed at {i}: {value} vs {reference}"
                        ));
                    }
                }
                println!(
                    "GPU integer matrix type {kind}, split {split}, rows {row_group}: {batch} vectors satisfy the f32 bound against the f64 reference"
                );
            }
        }
    }
    Ok(())
}
