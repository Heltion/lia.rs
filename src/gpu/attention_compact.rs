//! Independent checks for the proposed compact attention storage primitive.
#[cfg(test)]
mod tests {
    use crate::gpu::kernels;
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer};

    #[test]
    fn compact_qk_tail_matches_original_and_f64_with_padding_guards() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        for (batch, qh, kvh, start, count, tokens, key_start) in [
            (1usize, 4usize, 2usize, 0usize, 1usize, 33usize, 32usize),
            (65, 4, 2, 3, 59, 129, 96),
            (129, 16, 2, 16, 8, 513, 480),
        ] {
            let q = (0..batch * qh * 256)
                .map(|i| ((i * 17 % 37) as f32 - 18.0) * 0.2)
                .collect::<Vec<_>>();
            let k = (0..tokens * kvh * 256)
                .map(|i| (((i * 23 % 43) as f32 - 21.0) * 0.15) as f16)
                .collect::<Vec<_>>();
            let dq = DeviceBuffer::from_host(&stream, &q)?;
            let dk = DeviceBuffer::from_host(
                &stream,
                &k.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            )?;
            for grouped in [false, true] {
                let head = if grouped { kvh - 1 } else { qh - 1 };
                let sentinel = f32::from_bits(0x3f12_3456);
                let blank = vec![sentinel; 2048 + 160 + 17];
                let old = DeviceBuffer::from_host(&stream, &blank)?;
                let new = DeviceBuffer::from_host(&stream, &blank)?;
                macro_rules! launch {
                    ($compact:literal,$grouped:literal,$out:expr) => {{
                        let plan = module.flash_compact.prepare_qk::<$compact, $grouped>(
                            cuda_core::simt::LaunchConfig1D::new(
                                1,
                                512,
                                kernels::flash_compact::SHARED_BYTES,
                            ),
                        )?;
                        unsafe {
                            module.flash_compact.qk::<$compact, $grouped>(
                                &stream,
                                &plan,
                                qh,
                                kvh,
                                head,
                                start,
                                count,
                                key_start,
                                tokens,
                                &dq,
                                &dk,
                                $out.cu_deviceptr() as *mut f32,
                            )?;
                        }
                    }};
                }
                if grouped {
                    launch!(false, true, old);
                    launch!(true, true, new);
                } else {
                    launch!(false, false, old);
                    launch!(true, false, new);
                }
                let a = old.to_host_vec(&stream)?;
                let b = new.to_host_vec(&stream)?;
                assert_eq!(
                    a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
                for i in 0..160 {
                    assert_eq!(b[2048 + i].to_bits(), 0x3f12_0000 + i as u32);
                }
                assert!(b[2208..].iter().all(|x| x.to_bits() == sentinel.to_bits()));
                let groups = if grouped { qh / kvh } else { 1 };
                for row in 0..64 {
                    for col in 0..32 {
                        let query = start * groups + row;
                        let h = if grouped {
                            head * groups + query % groups
                        } else {
                            head
                        };
                        let kh = if grouped { head } else { head / (qh / kvh) };
                        let mut sum = 0.0f64;
                        let mut norm = 0.0f64;
                        if query < (start + count) * groups && key_start + col < tokens {
                            for c in 0..256 {
                                let p = q[((query / groups) * qh + h) * 256 + c] as f64
                                    * k[((key_start + col) * kvh + kh) * 256 + c] as f64
                                    / 16.0;
                                sum += p;
                                norm += p.abs();
                            }
                        }
                        assert!(
                            (b[row * 32 + col] as f64 - sum).abs() < 2e-6 * norm.max(1.0),
                            "row={row}, col={col}, grouped={grouped}"
                        );
                    }
                }
            }
        }
        Ok(())
    }
    #[test]
    fn compact_pv_survives_early_probability_overwrite() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        let probabilities = (0..32 * 32)
            .map(|i| ((i * 17 % 37) + 1) as f32 / 63.0)
            .collect::<Vec<_>>();
        let values = (0..32 * 256)
            .map(|i| (((i * 23 % 43) as f32 - 21.0) * 0.15) as f16)
            .collect::<Vec<_>>();
        let p = DeviceBuffer::from_host(&stream, &probabilities)?;
        let v = DeviceBuffer::from_host(
            &stream,
            &values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        )?;
        let sentinel = f32::from_bits(0x3f12_3456);
        let blank = vec![sentinel; 8192 + 17];
        let old = DeviceBuffer::from_host(&stream, &blank)?;
        let new = DeviceBuffer::from_host(&stream, &blank)?;
        macro_rules! run {
            ($compact:literal,$out:expr) => {{
                let plan = module.flash_compact_pv.prepare_pv::<$compact>(
                    cuda_core::simt::LaunchConfig1D::new(
                        1,
                        256,
                        kernels::flash_compact::SHARED_BYTES,
                    ),
                )?;
                unsafe {
                    module.flash_compact_pv.pv::<$compact>(
                        &stream,
                        &plan,
                        &p,
                        &v,
                        $out.cu_deviceptr() as *mut f32,
                    )?;
                }
            }};
        }
        run!(false, old);
        run!(true, new);
        let a = old.to_host_vec(&stream)?;
        let b = new.to_host_vec(&stream)?;
        assert_eq!(
            a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert!(b[8192..].iter().all(|x| x.to_bits() == sentinel.to_bits()));
        for row in 0..32 {
            for col in 0..256 {
                let mut sum = 0.0f64;
                let mut norm = 0.0f64;
                for k in 0..32 {
                    let term = probabilities[row * 32 + k] as f64 * values[k * 256 + col] as f64;
                    sum += term;
                    norm += term.abs();
                }
                assert!((b[row * 256 + col] as f64 - sum).abs() < 2e-6 * norm.max(1.0));
            }
        }
        Ok(())
    }
}
