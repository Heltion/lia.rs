use super::*;

pub(super) fn run() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
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
        module.convolution.gdn_gates(
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
            module.delta.delta_net::<false>(
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
            module.delta.delta_net::<true>(
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
        module.delta.delta_net::<true>(
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
        module.delta128.delta_net_128(
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

    Ok(())
}
