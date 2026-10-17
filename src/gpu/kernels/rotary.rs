use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub unsafe fn norm_rope(
        dim: usize,
        heads: usize,
        stride: usize,
        rot: usize,
        pos: usize,
        eps: f32,
        base: f32,
        input: &[f32],
        weight: &[f32],
        out: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let head = tid / 32;
        let lane = tid % 32;
        if head < input.len() / stride {
            let mut sum = 0.0f32;
            let mut j = lane;
            while j < dim {
                let x = input[head * stride + j];
                sum += x * x;
                j += 32;
            }
            let mut delta = 16;
            while delta > 0 {
                sum += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum, delta);
                delta /= 2;
            }
            sum = cuda_device::warp::shuffle_f32_sync(0xffff_ffff, sum, 0);
            let scale = 1.0 / (sum / dim as f32 + eps).sqrt();
            j = lane;
            while j < dim {
                let x = input[head * stride + j] * scale * weight[j];
                let value = if j < rot {
                    let p = j % (rot / 2);
                    let angle =
                        (pos + head / heads) as f32 * base.powf(-2.0 * p as f32 / rot as f32);
                    let pair = if j < rot / 2 {
                        j + rot / 2
                    } else {
                        j - rot / 2
                    };
                    let y = input[head * stride + pair] * scale * weight[pair];
                    x * angle.cos()
                        + if j < rot / 2 {
                            -y * angle.sin()
                        } else {
                            y * angle.sin()
                        }
                } else {
                    x
                };
                unsafe {
                    *out.add(head * dim + j) = value;
                }
                j += 32;
            }
        }
    }

    #[kernel]
    pub unsafe fn rope_table(rot: usize, batch: usize, position: usize, base: f32, out: *mut f32) {
        let i = thread::index_1d().get();
        let t = i / (rot / 2);
        let p = i % (rot / 2);
        if t < batch {
            let angle = (position + t) as f32 * base.powf(-2.0 * p as f32 / rot as f32);
            unsafe {
                *out.add(t * rot + p) = angle.cos();
                *out.add(t * rot + rot / 2 + p) = angle.sin();
            }
        }
    }

    #[kernel]
    pub unsafe fn norm_rope_cached(
        dim: usize,
        heads: usize,
        stride: usize,
        rot: usize,
        eps: f32,
        input: &[f32],
        weight: &[f32],
        rotary: &[f32],
        out: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let head = tid / 32;
        let lane = tid % 32;
        if head < input.len() / stride {
            let mut sum = 0.0f32;
            let mut j = lane;
            while j < dim {
                let x = input[head * stride + j];
                sum += x * x;
                j += 32;
            }
            let mut delta = 16;
            while delta > 0 {
                sum += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum, delta);
                delta /= 2;
            }
            sum = cuda_device::warp::shuffle_f32_sync(0xffff_ffff, sum, 0);
            let scale = 1.0 / (sum / dim as f32 + eps).sqrt();
            j = lane;
            while j < dim {
                let x = input[head * stride + j] * scale * weight[j];
                let value = if j < rot {
                    let p = j % (rot / 2);
                    let cos = rotary[(head / heads) * rot + p];
                    let sin = rotary[(head / heads) * rot + rot / 2 + p];
                    let pair = if j < rot / 2 {
                        j + rot / 2
                    } else {
                        j - rot / 2
                    };
                    let y = input[head * stride + pair] * scale * weight[pair];
                    x * cos + if j < rot / 2 { -y * sin } else { y * sin }
                } else {
                    x
                };
                unsafe {
                    *out.add(head * dim + j) = value;
                }
                j += 32;
            }
        }
    }

    #[kernel]
    pub unsafe fn cache_kv(
        dim: usize,
        pos: usize,
        k: &[f32],
        v: &[f32],
        kc: *mut f32,
        vc: *mut f32,
    ) {
        let i = thread::index_1d().get();
        if i < k.len() {
            unsafe {
                *kc.add(pos * dim + i) = k[i];
                *vc.add(pos * dim + i) = v[i];
            }
        }
    }

    #[kernel]
    pub unsafe fn cache_kv_half(
        dim: usize,
        position: usize,
        k: &[f32],
        v: &[f32],
        kc: *mut u16,
        vc: *mut u16,
    ) {
        let i = thread::index_1d().get();
        if i < k.len() {
            unsafe {
                *kc.add(position * dim + i) = (k[i] as f16).to_bits();
                *vc.add(position * dim + i) = (v[i] as f16).to_bits();
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
    fn cached_rotary() -> Result<()> {
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
}
