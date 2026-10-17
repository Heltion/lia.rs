use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub unsafe fn conv_silu(
        channels: usize,
        input: &[f32],
        weights: &[f32],
        history: *mut f32,
        captured: *mut f32,
        out: *mut f32,
    ) {
        let i = thread::index_1d().get();
        if i < channels {
            unsafe {
                let base = i * 3;
                let mut t = 0;
                while t < input.len() / channels {
                    let a = *history.add(base);
                    let b = *history.add(base + 1);
                    let c = *history.add(base + 2);
                    let x = input[t * channels + i];
                    let value = a * weights[i * 4]
                        + b * weights[i * 4 + 1]
                        + c * weights[i * 4 + 2]
                        + x * weights[i * 4 + 3];
                    *history.add(base) = b;
                    *history.add(base + 1) = c;
                    *history.add(base + 2) = x;
                    if !captured.is_null() {
                        *captured.add(t * channels * 3 + base) = b;
                        *captured.add(t * channels * 3 + base + 1) = c;
                        *captured.add(t * channels * 3 + base + 2) = x;
                    }
                    *out.add(t * channels + i) = value / (1.0 + (-value).exp());
                    t += 1;
                }
            }
        }
    }

    #[kernel]
    pub unsafe fn l2_norm(
        dim: usize,
        groups: usize,
        per_token: usize,
        stride: usize,
        eps: f32,
        x: &[f32],
        out: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        if group < groups {
            let mut sum = 0.0f32;
            let mut j = lane;
            while j < dim {
                let v = x[(group / per_token) * stride + (group % per_token) * dim + j];
                sum += v * v;
                j += 32;
            }
            let mut delta = 16;
            while delta > 0 {
                sum += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum, delta);
                delta /= 2;
            }
            sum = cuda_device::warp::shuffle_f32_sync(0xffff_ffff, sum, 0);
            let scale = 1.0 / (sum + eps).sqrt();
            j = lane;
            while j < dim {
                unsafe {
                    *out.add(group * dim + j) =
                        x[(group / per_token) * stride + (group % per_token) * dim + j] * scale;
                }
                j += 32;
            }
        }
    }

    #[kernel]
    pub unsafe fn gdn_gates(
        heads: usize,
        alpha: &[f32],
        beta: &[f32],
        dt: &[f32],
        a: &[f32],
        decay: *mut f32,
        bnorm: *mut f32,
    ) {
        let i = thread::index_1d().get();
        if i < alpha.len() {
            let h = i % heads;
            let x = alpha[i] + dt[h];
            let softplus = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
            unsafe {
                *decay.add(i) = (a[h] * softplus).exp();
                *bnorm.add(i) = 1.0 / (1.0 + (-beta[i]).exp());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::gpu::{kernels, tests::close};
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};

    #[test]
    fn convolution() -> Result<()> {
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
}
