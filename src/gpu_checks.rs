//! Independent small f64 references for stateful GPU operator validation.
use crate::gpu::kernels;
use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn close(actual: &[f32], expected: &[f64], name: &str) -> Result<()> {
    if actual.len() != expected.len() {
        return Err(format!("{name}: length mismatch").into());
    }
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || (*a as f64 - e).abs() > 2e-5 * e.abs().max(1.0) {
            return Err(format!("{name}[{i}]: {a} != {e}").into());
        }
    }
    Ok(())
}
pub fn run() -> Result<()> {
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
        module.bf16_partial(
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
        module.bf16_merge(
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
        module.argmax_partial(
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
            return Err("GPU greedy reduction disagrees on ties or signed zero".into());
        }
    }
    println!(
        "GPU vocabulary reduction matches total float order, partition boundaries and last-index ties"
    );
    let channels = 5;
    let batch = 4;
    let input: Vec<f32> = (0..channels * batch)
        .map(|i| (i as f32 - 9.0) * 0.1)
        .collect();
    let weights: Vec<f32> = (0..channels * 4).map(|i| (i as f32 - 7.0) * 0.03).collect();
    let history = DeviceBuffer::<f32>::zeroed(&stream, channels * 3)?;
    let captured = DeviceBuffer::<f32>::zeroed(&stream, batch * channels * 3)?;
    let output = DeviceBuffer::<f32>::zeroed(&stream, input.len())?;
    let di = DeviceBuffer::from_host(&stream, &input)?;
    let dw = DeviceBuffer::from_host(&stream, &weights)?;
    // SAFETY: every channel has three history values and four filter taps;
    // captured history has one complete allocation per input token.
    unsafe {
        module.conv_silu(
            &stream,
            LaunchConfig::for_num_elems(channels as u32),
            channels,
            &di,
            &dw,
            history.cu_deviceptr() as *mut f32,
            captured.cu_deviceptr() as *mut f32,
            output.cu_deviceptr() as *mut f32,
        )?;
    }
    let mut expected_conv = Vec::new();
    let mut expected_capture = Vec::new();
    let mut h = vec![0.0f64; channels * 3];
    for t in 0..batch {
        for i in 0..channels {
            let x = input[t * channels + i] as f64;
            let value = h[i * 3] * weights[i * 4] as f64
                + h[i * 3 + 1] * weights[i * 4 + 1] as f64
                + h[i * 3 + 2] * weights[i * 4 + 2] as f64
                + x * weights[i * 4 + 3] as f64;
            expected_conv.push(value / (1.0 + (-value).exp()));
            h[i * 3] = h[i * 3 + 1];
            h[i * 3 + 1] = h[i * 3 + 2];
            h[i * 3 + 2] = x;
        }
        expected_capture.extend_from_slice(&h);
    }
    close(
        &output.to_host_vec(&stream)?,
        &expected_conv,
        "batched convolution",
    )?;
    close(
        &captured.to_host_vec(&stream)?,
        &expected_capture,
        "captured convolution states",
    )?;
    println!("GPU batched convolution and every captured state match f64 reference");
    let dim = 128;
    let kh = 2;
    let vh = 4;
    let qk: Vec<f32> = (0..2 * kh * dim)
        .map(|i| ((i * 17 % 29) as f32 - 14.0) * 0.02)
        .collect();
    let conv: Vec<f32> = (0..(2 * kh + vh) * dim)
        .map(|i| ((i * 11 % 31) as f32 - 15.0) * 0.03)
        .collect();
    let alpha = [-1.3f32, 0.0, 0.7, 25.0];
    let beta = [-2.0f32, 0.5, -0.1, 1.7];
    let dt = [0.2f32, -0.1, 0.5, 0.3];
    let a = [-0.3f32, -0.8, -1.2, -0.1];
    let mut expected: Vec<f64> = (0..vh * dim * dim)
        .map(|i| ((i * 7 % 13) as f64 - 6.0) * 0.001)
        .collect();
    let initial: Vec<f32> = expected.iter().map(|x| *x as f32).collect();
    let dqk = DeviceBuffer::from_host(&stream, &qk)?;
    let dc = DeviceBuffer::from_host(&stream, &conv)?;
    let da = DeviceBuffer::from_host(&stream, &alpha)?;
    let db = DeviceBuffer::from_host(&stream, &beta)?;
    let ddt = DeviceBuffer::from_host(&stream, &dt)?;
    let dscale = DeviceBuffer::from_host(&stream, &a)?;
    let state = DeviceBuffer::from_host(&stream, &initial)?;
    let optimized_state = DeviceBuffer::from_host(&stream, &initial)?;
    let decay = DeviceBuffer::<f32>::zeroed(&stream, vh)?;
    let bnorm = DeviceBuffer::<f32>::zeroed(&stream, vh)?;
    // SAFETY: each head has one producer and all gate vectors have vh elements.
    unsafe {
        module.gdn_gates(
            &stream,
            LaunchConfig::for_num_elems(vh as u32),
            vh,
            &da,
            &db,
            &ddt,
            &dscale,
            decay.cu_deviceptr() as *mut f32,
            bnorm.cu_deviceptr() as *mut f32,
        )?;
    }

    let out = DeviceBuffer::<f32>::zeroed(&stream, vh * dim)?;
    let mut expected_history = Vec::new();
    for step in 0..8 {
        let mut expected_out = vec![0f64; vh * dim];
        for h in 0..vh {
            let x = alpha[h] as f64 + dt[h] as f64;
            let softplus = if x > 20.0 { x } else { x.exp().ln_1p() };
            let decay = (a[h] as f64 * softplus).exp();
            let b = 1.0 / (1.0 + (-beta[h] as f64).exp());
            for v in 0..dim {
                let row = h * dim + v;
                let dot: f64 = (0..dim)
                    .map(|j| expected[row * dim + j] * qk[(kh + h % kh) * dim + j] as f64)
                    .sum();
                let correction = (conv[2 * kh * dim + h * dim + v] as f64 - decay * dot) * b;
                for j in 0..dim {
                    expected[row * dim + j] = decay * expected[row * dim + j]
                        + qk[(kh + h % kh) * dim + j] as f64 * correction;
                    expected_out[row] += expected[row * dim + j] * qk[(h % kh) * dim + j] as f64
                        / (dim as f64).sqrt();
                }
            }
        }
        // SAFETY: complete warps, validated small input shapes and one warp per
        // disjoint state row. Stream ordering separates successive updates.
        unsafe {
            module.delta_net::<false>(
                &stream,
                LaunchConfig::for_num_elems((vh * dim * 32) as u32),
                dim,
                kh,
                vh,
                &dqk,
                &dc,
                &da,
                &db,
                &ddt,
                &dscale,
                state.cu_deviceptr() as *mut f32,
                core::ptr::null_mut(),
                out.cu_deviceptr() as *mut f32,
            )?;
        }
        // SAFETY: same complete state rows as the reference kernel, with the
        // per-head gate values computed once by the preceding launch.
        unsafe {
            module.delta_net::<true>(
                &stream,
                LaunchConfig::for_num_elems((vh * dim * 32) as u32),
                dim,
                kh,
                vh,
                &dqk,
                &dc,
                &decay,
                &bnorm,
                &ddt,
                &dscale,
                optimized_state.cu_deviceptr() as *mut f32,
                core::ptr::null_mut(),
                out.cu_deviceptr() as *mut f32,
            )?;
        }
        close(
            &optimized_state.to_host_vec(&stream)?,
            &expected,
            "precomputed-gate delta state",
        )?;
        close(&state.to_host_vec(&stream)?, &expected, "delta state")?;
        close(&out.to_host_vec(&stream)?, &expected_out, "delta output")?;
        expected_history.extend_from_slice(&expected);
        println!("GPU gated delta recurrence: step {step} state and output match f64 reference");
    }
    let batched_state = DeviceBuffer::from_host(&stream, &initial)?;
    let captured_states = DeviceBuffer::<f32>::zeroed(&stream, 8 * initial.len())?;
    let batched_out = DeviceBuffer::<f32>::zeroed(&stream, 8 * vh * dim)?;
    let bq = DeviceBuffer::from_host(&stream, &qk.repeat(8))?;
    let bc = DeviceBuffer::from_host(&stream, &conv.repeat(8))?;
    let bd = DeviceBuffer::from_host(&stream, &decay.to_host_vec(&stream)?.repeat(8))?;
    let bb = DeviceBuffer::from_host(&stream, &bnorm.to_host_vec(&stream)?.repeat(8))?;
    // SAFETY: eight complete input positions, disjoint state rows, and eight
    // complete captured matrices. Gates are precomputed exactly as above.
    unsafe {
        module.delta_net::<true>(
            &stream,
            LaunchConfig::for_num_elems((vh * dim * 32) as u32),
            dim,
            kh,
            vh,
            &bq,
            &bc,
            &bd,
            &bb,
            &ddt,
            &dscale,
            batched_state.cu_deviceptr() as *mut f32,
            captured_states.cu_deviceptr() as *mut f32,
            batched_out.cu_deviceptr() as *mut f32,
        )?;
    }
    close(
        &captured_states.to_host_vec(&stream)?,
        &expected_history,
        "captured batched recurrence",
    )?;
    println!("GPU speculative recurrence captures all eight intermediate states correctly");
    let register_state = DeviceBuffer::from_host(&stream, &initial)?;
    let register_capture = DeviceBuffer::<f32>::zeroed(&stream, 8 * initial.len())?;
    // SAFETY: 128-wide rows with four elements per lane and eight complete
    // token states; this uses the same independent f64 history reference.
    unsafe {
        module.delta_net_128(
            &stream,
            LaunchConfig::for_num_elems((vh * dim * 32) as u32),
            kh,
            vh,
            &bq,
            &bc,
            &bd,
            &bb,
            register_state.cu_deviceptr() as *mut f32,
            register_capture.cu_deviceptr() as *mut f32,
            batched_out.cu_deviceptr() as *mut f32,
        )?;
    }
    close(
        &register_capture.to_host_vec(&stream)?,
        &expected_history,
        "register-resident recurrent history",
    )?;
    close(
        &register_state.to_host_vec(&stream)?,
        &expected,
        "register-resident final state",
    )?;
    println!("GPU register-resident recurrence matches every independent f64 intermediate state");

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
        module.attn_scores(
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
        module.softmax(
            &stream,
            LaunchConfig::for_num_elems((qh * 32) as u32),
            qh,
            tokens,
            capacity,
            scores.cu_deviceptr() as *mut f32,
        )?;
        module.attn_values(
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
        module.flash_partial::<8, 2>(
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
        module.flash_merge::<8>(
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
            module.quantize_activation(
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
            return Err("GPU activation quantization mismatch".into());
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
                        return Err(format!(
                            "integer matrix rounding bound failed at {i}: {value} vs {reference}"
                        )
                        .into());
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
        module.flash_partial::<256, 128>(
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
        module.flash_merge::<256>(
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
        module.cache_kv_half(
            &stream,
            LaunchConfig::for_num_elems(k.len() as u32),
            kvh * dim,
            0,
            &dk,
            &dv,
            k16.cu_deviceptr() as *mut u16,
            v16.cu_deviceptr() as *mut u16,
        )?;
        module.flash_partial_half::<256, 128>(
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
        module.flash_merge::<256>(
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
            module.rope_table(
                &stream,
                LaunchConfig::for_num_elems((batch * rot / 2) as u32),
                rot,
                batch,
                position,
                base,
                table.cu_deviceptr() as *mut f32,
            )?;
            module.norm_rope(
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
            module.norm_rope_cached(
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
