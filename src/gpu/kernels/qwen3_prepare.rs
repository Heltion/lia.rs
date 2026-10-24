//! One warp per Q/K head preserves RMS reduction and split-half RoPE order.
use cuda_device::{kernel, thread, warp};
use cuda_host::cuda_module;
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub unsafe fn qwen3_prepare(
        batch: usize,
        position: usize,
        base: f32,
        eps: f32,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        qw: &[f32],
        kw: &[f32],
        output: *mut f32,
        kc: *mut u16,
        vc: *mut u16,
    ) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        if group < batch * 40 {
            let token = group / 40;
            let head = group % 40;
            let query = head < 32;
            let offset = if query {
                token * 4096 + head * 128
            } else {
                token * 1024 + (head - 32) * 128
            };
            let x = if query { q } else { k };
            let weight = if query { qw } else { kw };
            let mut sum = 0.0f32;
            let mut j = lane;
            while j < 128 {
                let a = x[offset + j];
                sum += a * a;
                j += 32;
            }
            let mut delta = 16;
            while delta > 0 {
                sum += warp::shuffle_down_f32_sync(0xffff_ffff, sum, delta);
                delta /= 2;
            }
            sum = warp::shuffle_f32_sync(0xffff_ffff, sum, 0);
            let scale = 1.0 / (sum / 128.0 + eps).sqrt();
            j = lane;
            while j < 128 {
                let paired = if j < 64 { j + 64 } else { j - 64 };
                let a = x[offset + j] * scale * weight[j];
                let b = x[offset + paired] * scale * weight[paired];
                let angle =
                    (position + token) as f32 * (1.0 / base.powf(2.0 * (j % 64) as f32 / 128.0));
                let value = a * angle.cos() + b * angle.sin() * if j < 64 { -1.0 } else { 1.0 };
                unsafe {
                    if query {
                        *output.add(offset + j) = value;
                    } else {
                        *kc.add(position * 1024 + offset + j) = (value as f16).to_bits();
                        *vc.add(position * 1024 + offset + j) = (v[offset + j] as f16).to_bits();
                    }
                }
                j += 32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::gpu::kernels;
    use anyhow::{Result, ensure};
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};

    #[test]
    fn fused_head_preparation_matches_separate_operations() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        for (position, batch) in [(0usize, 1usize), (31, 17), (4090, 65), (129, 512)] {
            let values = |n: usize, seed: usize| -> Vec<f32> {
                (0..n)
                    .map(|i| ((i * 17 + seed) % 257) as f32 * 0.0137 - 1.73)
                    .collect()
            };
            let q = DeviceBuffer::from_host(&stream, &values(batch * 4096, 1))?;
            let k = DeviceBuffer::from_host(&stream, &values(batch * 1024, 7))?;
            let v = DeviceBuffer::from_host(&stream, &values(batch * 1024, 19))?;
            let weights: Vec<f32> = (0..128).map(|i| 0.3 + i as f32 * 0.0173).collect();
            let qw = DeviceBuffer::from_host(&stream, &weights)?;
            let kw = DeviceBuffer::from_host(
                &stream,
                &weights.iter().rev().copied().collect::<Vec<_>>(),
            )?;
            let qn = DeviceBuffer::zeroed(&stream, q.len())?;
            let kn = DeviceBuffer::zeroed(&stream, k.len())?;
            let kr = DeviceBuffer::zeroed(&stream, k.len())?;
            let guard = 12345.0f32;
            let expected = DeviceBuffer::from_host(&stream, &vec![guard; q.len() + 17])?;
            let actual = DeviceBuffer::from_host(&stream, &vec![guard; q.len() + 17])?;
            let cache_n = (position + batch) * 1024 + 17;
            let cache = || DeviceBuffer::from_host(&stream, &vec![0x55aau16; cache_n]);
            let ek = cache()?;
            let ev = cache()?;
            let ak = cache()?;
            let av = cache()?;
            let positions = DeviceBuffer::zeroed(&stream, 1)?;
            // SAFETY: complete 128-value heads, bounded positions and disjoint
            // logical outputs with extra sentinel regions for every final store.
            unsafe {
                module.normalization.rms_norm(
                    &stream,
                    LaunchConfig::for_num_elems((batch * 32 * 32) as u32),
                    128,
                    batch * 32,
                    1e-6,
                    &q,
                    &qw,
                    qn.cu_deviceptr() as *mut f32,
                )?;
                module.normalization.rms_norm(
                    &stream,
                    LaunchConfig::for_num_elems((batch * 8 * 32) as u32),
                    128,
                    batch * 8,
                    1e-6,
                    &k,
                    &kw,
                    kn.cu_deviceptr() as *mut f32,
                )?;
                module.llama.llama_rope(
                    &stream,
                    LaunchConfig::for_num_elems(q.len() as u32),
                    32,
                    position,
                    1e6,
                    false,
                    &positions,
                    &qn,
                    expected.cu_deviceptr() as *mut f32,
                )?;
                module.llama.llama_rope(
                    &stream,
                    LaunchConfig::for_num_elems(k.len() as u32),
                    8,
                    position,
                    1e6,
                    false,
                    &positions,
                    &kn,
                    kr.cu_deviceptr() as *mut f32,
                )?;
                module.dense_attention.dense_store_kv(
                    &stream,
                    LaunchConfig::for_num_elems(k.len() as u32),
                    position * 1024,
                    &kr,
                    &v,
                    ek.cu_deviceptr() as *mut u16,
                    ev.cu_deviceptr() as *mut u16,
                )?;
                module.qwen3_prepare.qwen3_prepare(
                    &stream,
                    LaunchConfig::for_num_elems((batch * 40 * 32) as u32),
                    batch,
                    position,
                    1e6,
                    1e-6,
                    &q,
                    &k,
                    &v,
                    &qw,
                    &kw,
                    actual.cu_deviceptr() as *mut f32,
                    ak.cu_deviceptr() as *mut u16,
                    av.cu_deviceptr() as *mut u16,
                )?;
            }
            let actual = actual.to_host_vec(&stream)?;
            let expected = expected.to_host_vec(&stream)?;
            let mut maximum = 0.0f32;
            let mut differences = 0;
            for (a, b) in actual.iter().zip(&expected) {
                maximum = maximum.max((*a - b).abs());
                differences += usize::from(a.to_bits() != b.to_bits());
            }
            ensure!(
                differences == 0,
                "fusedQ position={position} batch={batch}: differences={differences}, maximum={maximum}"
            );
            ensure!(
                ak.to_host_vec(&stream)? == ek.to_host_vec(&stream)?,
                "fused K cache mismatch"
            );
            ensure!(
                av.to_host_vec(&stream)? == ev.to_host_vec(&stream)?,
                "fused V cache mismatch"
            );
            println!(
                "Qwen3 fusedprepare position={position} batch={batch}: Q/K/V bitexact, guards intact"
            );
        }
        Ok(())
    }
}
