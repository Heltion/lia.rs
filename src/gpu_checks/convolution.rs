use super::*;

pub(super) fn run() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
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
        module.convolution.conv_silu(
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
    Ok(())
}
