//! BF16 KV storage conversion with the same positional ABI as the FP16 writer.
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub unsafe fn cache_kv_bf16(
        dim: usize,
        position: usize,
        k: &[f32],
        v: &[f32],
        kc: *mut u16,
        vc: *mut u16,
    ) {
        let i = thread::index_1d().get() * 2;
        if i >= k.len() {
            return;
        }
        let (kn, vn) = if i + 1 < k.len() {
            (k[i + 1], v[i + 1])
        } else {
            (0.0, 0.0)
        };
        let a = cuda_device::convert::cvt_bf16x2_f32(k[i], kn);
        let b = cuda_device::convert::cvt_bf16x2_f32(v[i], vn);
        // SAFETY: complete matching input extents and positional cache capacity
        // are host validated. Halfword stores also support odd-row offsets.
        unsafe {
            *kc.add(position * dim + i) = a as u16;
            *vc.add(position * dim + i) = b as u16;
            if i + 1 < k.len() {
                *kc.add(position * dim + i + 1) = (a >> 16) as u16;
                *vc.add(position * dim + i + 1) = (b >> 16) as u16;
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};
    #[test]
    fn bf16_cache_rounding_offsets_tails_and_guards() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = super::super::load(&context)?;
        let edge = [
            0.0,
            -0.0,
            1.00390625,
            1.01171875,
            -1.00390625,
            f32::MIN_POSITIVE,
            f32::MAX,
            -f32::MAX,
            f32::from_bits(1),
            1.0 / 31.0,
        ];
        let round = |x: f32| {
            let b = x.to_bits();
            (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
        };
        for (dim, batch, position) in [
            (1usize, 1usize, 0usize),
            (65, 3, 1),
            (512, 1, 7),
            (512, 9, 3),
        ] {
            let k = (0..dim * batch)
                .map(|i| {
                    if i % 3 == 0 {
                        edge[i % edge.len()]
                    } else {
                        ((i * 71 % 4093) as f32 - 2046.0) / 31.0
                    }
                })
                .collect::<Vec<_>>();
            let v = (0..dim * batch)
                .map(|i| k[k.len() - 1 - i])
                .collect::<Vec<_>>();
            let dk = DeviceBuffer::from_host(&stream, &k)?;
            let dv = DeviceBuffer::from_host(&stream, &v)?;
            let offset = position * dim;
            let kc = DeviceBuffer::from_host(&stream, &vec![0x7badu16; offset + k.len() + 17])?;
            let vc = DeviceBuffer::from_host(&stream, &vec![0x7badu16; offset + k.len() + 17])?;
            // SAFETY: guarded allocations cover every input and positional row.
            unsafe {
                module.bf16_cache.cache_kv_bf16(
                    &stream,
                    LaunchConfig::for_num_elems(k.len().div_ceil(2) as u32),
                    dim,
                    position,
                    &dk,
                    &dv,
                    kc.cu_deviceptr() as *mut u16,
                    vc.cu_deviceptr() as *mut u16,
                )?;
            }
            for (actual, input) in [
                (kc.to_host_vec(&stream)?, &k),
                (vc.to_host_vec(&stream)?, &v),
            ] {
                assert!(actual[..offset].iter().all(|x| *x == 0x7bad));
                assert!(actual[offset + input.len()..].iter().all(|x| *x == 0x7bad));
                assert_eq!(
                    &actual[offset..offset + input.len()],
                    input.iter().map(|x| round(*x)).collect::<Vec<_>>()
                );
            }
        }
        Ok(())
    }
}
