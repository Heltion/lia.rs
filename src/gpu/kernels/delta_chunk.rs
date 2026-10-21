//! Chunked gated delta updates. Recurrent state and persistent accumulators stay f32.
use super::delta_mma::{load_state, product, put, scalar, store_state};
use super::delta_tiles::{BETA, LOWER, PREFIX, RATIO, invert, matrices, prepare, prepare_qk};
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[kernel]
    pub unsafe fn delta_chunk<const PREPARED: bool>(
        k_heads: usize,
        v_heads: usize,
        qk: &[f32],
        conv: &[f32],
        decay: &[f32],
        beta: &[f32],
        prepared: &[u32],
        state: *mut f32,
        out: *mut f32,
    ) {
        static mut TILE: cuda_device::SharedArray<u32, 11440, 16> =
            cuda_device::SharedArray::UNINIT;
        let tid = thread::index_1d().get() as u32;
        let local = tid % 256;
        let warp = local / 32;
        let lane = local % 32;
        let head = (tid / 256) / 4;
        let value_base = ((tid / 256) % 4) * 32;
        let k_heads = k_heads as u32;
        let v_heads = v_heads as u32;
        let batch = decay.len() as u32 / v_heads;
        // SAFETY: complete CTAs own disjoint head/value slices and reach every
        // barrier. The host supplies padded input extents and full state rows.
        let tile = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut TILE) };
        let [q, k, p, inverse, delta, scratch_state, solved] = unsafe { matrices(tile) };
        let mut carry = [[0.0f32; 4]; 4];
        let mut j = 0usize;
        #[unroll]
        while j < 4 {
            let fragment = warp + j as u32 * 8;
            carry[j] = unsafe { load_state(state, head, value_base, fragment, lane) };
            j += 1;
        }
        let mut start = 0;
        while start < batch {
            let valid = (batch - start).min(16);
            j = 0;
            #[unroll]
            while j < 4 {
                let fragment = warp + j as u32 * 8;
                unsafe {
                    scratch_state.fragment((fragment / 4) * 16, (fragment % 4) * 8, lane, carry[j]);
                }
                j += 1;
            }
            unsafe {
                if PREPARED {
                    prepare_qk(q, k, head, k_heads, start, valid, local, qk);
                    let base = (head * batch.div_ceil(16) + start / 16) * 1072;
                    let mut i = local;
                    while i < 1072 {
                        let dest = if i < 768 { 4352 + i } else { RATIO + i - 768 };
                        tile.add(dest as usize)
                            .write(*prepared.get_unchecked((base + i) as usize));
                        i += 256;
                    }
                    thread::sync_threads();
                } else {
                    prepare(
                        tile, q, k, head, k_heads, v_heads, start, valid, local, qk, decay, beta,
                    );
                }
            }
            let mut base_output = [0.0f32; 4];
            if warp < 4 && !PREPARED {
                let col = (warp % 2) * 8;
                let gram = warp < 2;
                let x = unsafe {
                    product::<128, false, false>(
                        [0.0; 4],
                        if gram { k } else { q },
                        k,
                        0,
                        col,
                        lane,
                    )
                };
                let mut coefficients = [0.0; 4];
                j = 0;
                #[unroll]
                while j < 4 {
                    let r = lane / 4 + (j as u32 / 2) * 8;
                    let c = col + (lane % 4) * 2 + j as u32 % 2;
                    let value = if r < valid && c <= r {
                        x[j] * unsafe { scalar(tile, RATIO + r * 16 + c) }
                    } else {
                        0.0
                    };
                    unsafe {
                        if gram {
                            put(
                                tile,
                                LOWER + r * 16 + c,
                                if c < r {
                                    value * scalar(tile, BETA + r)
                                } else {
                                    0.0
                                },
                            );
                        } else {
                            coefficients[j] = value;
                        }
                    }
                    j += 1;
                }
                if !gram {
                    unsafe {
                        p.fragment(0, col, lane, coefficients);
                    }
                }
            } else if warp >= 4 {
                let col = (warp - 4) * 8;
                let qs = unsafe {
                    product::<128, false, true>([0.0; 4], q, scratch_state, 0, col, lane)
                };
                let ks = unsafe {
                    product::<128, false, true>([0.0; 4], k, scratch_state, 0, col, lane)
                };
                let mut rhs = [0.0; 4];
                j = 0;
                #[unroll]
                while j < 4 {
                    let t = lane / 4 + (j as u32 / 2) * 8;
                    let v = col + (lane % 4) * 2 + j as u32 % 2;
                    unsafe {
                        let g = scalar(tile, PREFIX + t);
                        let vv = if t < valid {
                            *conv.get_unchecked(
                                ((start + t) * (2 * k_heads + v_heads) * 128
                                    + 2 * k_heads * 128
                                    + head * 128
                                    + value_base
                                    + v) as usize,
                            )
                        } else {
                            0.0
                        };
                        rhs[j] = scalar(tile, BETA + t) * (vv - g * ks[j]);
                        base_output[j] = g * qs[j];
                    }
                    j += 1;
                }
                unsafe {
                    delta.fragment(0, col, lane, rhs);
                }
            }
            thread::sync_threads();
            unsafe {
                if !PREPARED {
                    invert(tile, inverse, warp, lane);
                }
            }
            thread::sync_threads();
            if warp < 4 {
                let col = warp * 8;
                unsafe {
                    let x = product::<16, false, true>([0.0; 4], inverse, delta, 0, col, lane);
                    solved.fragment(0, col, lane, x);
                }
            }
            thread::sync_threads();
            if warp >= 4 {
                let col = (warp - 4) * 8;
                let x = unsafe { product::<16, false, true>([0.0; 4], p, solved, 0, col, lane) };
                j = 0;
                #[unroll]
                while j < 4 {
                    let t = lane / 4 + (j as u32 / 2) * 8;
                    let v = col + (lane % 4) * 2 + j as u32 % 2;
                    if t < valid {
                        unsafe {
                            *out.add(
                                ((start + t) * v_heads * 128 + head * 128 + value_base + v)
                                    as usize,
                            ) = (base_output[j] + x[j]) / 128.0f32.sqrt();
                        }
                    }
                    j += 1;
                }
            }
            let mut i = local * 2;
            while i < 512 {
                let t = i / 32;
                let v = i % 32;
                unsafe {
                    let (a, b) = solved.read_pair(t, v);
                    let factor = scalar(tile, RATIO + (valid - 1) * 16 + t);
                    delta.store_pair(t, v, a * factor, b * factor);
                }
                i += 512;
            }
            thread::sync_threads();
            let end_decay = unsafe { scalar(tile, PREFIX + valid - 1) };
            j = 0;
            #[unroll]
            while j < 4 {
                let fragment = warp + j as u32 * 8;
                let mut x = carry[j];
                x[0] *= end_decay;
                x[1] *= end_decay;
                x[2] *= end_decay;
                x[3] *= end_decay;
                carry[j] = unsafe {
                    product::<16, true, true>(
                        x,
                        k,
                        delta,
                        (fragment / 4) * 16,
                        (fragment % 4) * 8,
                        lane,
                    )
                };
                j += 1;
            }
            thread::sync_threads();
            start += 16;
        }
        j = 0;
        #[unroll]
        while j < 4 {
            let fragment = warp + j as u32 * 8;
            unsafe {
                store_state(state, head, value_base, fragment, lane, carry[j]);
            }
            j += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::gpu::{kernels, tests::close};
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};

    #[test]
    fn chunked_recurrence() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        let (kh, vh, dim) = (2, 4, 128);
        for batch in [1, 9, 16, 17, 33, 64, 512, 1024, 2048] {
            let qk: Vec<f32> = (0..batch * 2 * kh * dim)
                .map(|i| ((i * 17 % 37) as f32 - 18.0) * 0.006)
                .collect();
            let conv: Vec<f32> = (0..batch * (2 * kh + vh) * dim)
                .map(|i| ((i * 11 % 31) as f32 - 15.0) * 0.03)
                .collect();
            let decay: Vec<f32> = (0..batch * vh)
                .map(|i| {
                    if i / vh == 6 {
                        0.0
                    } else if i / vh == 23 {
                        1e-40
                    } else {
                        0.995 - (i % 5) as f32 * 0.1
                    }
                })
                .collect();
            let beta: Vec<f32> = (0..batch * vh)
                .map(|i| 0.2 + (i % 7) as f32 * 0.1)
                .collect();
            let initial: Vec<f32> = (0..vh * dim * dim)
                .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.002)
                .collect();
            let mut expected: Vec<f64> = initial.iter().map(|x| *x as f64).collect();
            let mut expected_out = vec![0.0f64; batch * vh * dim];
            for t in 0..batch {
                for h in 0..vh {
                    let qo = t * 2 * kh * dim + (h % kh) * dim;
                    let ko = qo + kh * dim;
                    let a = decay[t * vh + h] as f64;
                    let b = beta[t * vh + h] as f64;
                    for v in 0..dim {
                        let row = (h * dim + v) * dim;
                        let dot: f64 = (0..dim)
                            .map(|d| expected[row + d] * qk[ko + d] as f64)
                            .sum();
                        let correction = b
                            * (conv[t * (2 * kh + vh) * dim + 2 * kh * dim + h * dim + v] as f64
                                - a * dot);
                        let mut output = 0.0;
                        for d in 0..dim {
                            expected[row + d] =
                                a * expected[row + d] + qk[ko + d] as f64 * correction;
                            output += expected[row + d] * qk[qo + d] as f64;
                        }
                        expected_out[t * vh * dim + h * dim + v] = output / (dim as f64).sqrt();
                    }
                }
            }
            let qk = DeviceBuffer::from_host(&stream, &qk)?;
            let conv = DeviceBuffer::from_host(&stream, &conv)?;
            let decay = DeviceBuffer::from_host(&stream, &decay)?;
            let beta = DeviceBuffer::from_host(&stream, &beta)?;
            let prepared = DeviceBuffer::<u32>::zeroed(&stream, vh * batch.div_ceil(16) * 1072)?;
            for parallel in [false, true] {
                let state = DeviceBuffer::from_host(&stream, &initial)?;
                let out = DeviceBuffer::<f32>::zeroed(&stream, batch * vh * dim)?;
                // SAFETY: one complete256-thread CTA per32-value slice, full
                //128-wide state rows, and complete padded-input/output extents.
                unsafe {
                    if parallel {
                        module.delta_prepare.prepare_chunks(
                            &stream,
                            LaunchConfig {
                                grid_dim: ((vh * batch.div_ceil(16)) as u32, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            kh,
                            vh,
                            &qk,
                            &decay,
                            &beta,
                            prepared.cu_deviceptr() as *mut u32,
                        )?;
                    }
                    macro_rules! run {
                        ($ready:literal) => {
                            module.delta_chunk.delta_chunk::<$ready>(
                                &stream,
                                LaunchConfig {
                                    grid_dim: ((vh * 4) as u32, 1, 1),
                                    block_dim: (256, 1, 1),
                                    shared_mem_bytes: 0,
                                },
                                kh,
                                vh,
                                &qk,
                                &conv,
                                &decay,
                                &beta,
                                &prepared,
                                state.cu_deviceptr() as *mut f32,
                                out.cu_deviceptr() as *mut f32,
                            )?
                        };
                    }
                    if parallel {
                        run!(true);
                    } else {
                        run!(false);
                    }
                }
                close(&out.to_host_vec(&stream)?, &expected_out, "chunked outputs")?;
                close(
                    &state.to_host_vec(&stream)?,
                    &expected,
                    "chunked final state",
                )?;
                println!(
                    "chunk batch={batch} parallel={parallel}: all outputs and final state match independent f64 recurrence"
                );
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires LIA_TEST_GDN_REFERENCE with external real-input float64 reference"]
    fn real_input_recurrence_and_partition() -> Result<()> {
        use anyhow::{Context, ensure};
        use std::path::{Path, PathBuf};
        fn floats(path: &Path) -> Result<Vec<f32>> {
            let bytes =
                std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
            ensure!(
                bytes.len().is_multiple_of(4),
                "invalid f32 reference length"
            );
            Ok(bytes
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect())
        }
        fn doubles(path: &Path) -> Result<Vec<f64>> {
            let bytes =
                std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
            ensure!(
                bytes.len().is_multiple_of(8),
                "invalid f64 reference length"
            );
            Ok(bytes
                .chunks_exact(8)
                .map(|v| f64::from_le_bytes(v.try_into().unwrap()))
                .collect())
        }
        fn compare(actual: &[f32], expected: &[f64], name: &str) -> Result<()> {
            ensure!(actual.len() == expected.len(), "reference shape mismatch");
            let mut error = 0.0f64;
            let mut norm = 0.0f64;
            let mut maximum = 0.0f64;
            for (&a, &b) in actual.iter().zip(expected) {
                let delta = f64::from(a) - b;
                error += delta * delta;
                norm += b * b;
                maximum = maximum.max(delta.abs());
            }
            eprintln!(
                "real GDN {name} relative_l2={} max_abs={maximum}",
                (error / norm).sqrt()
            );
            close(actual, expected, name)
        }
        let directory = PathBuf::from(
            std::env::var("LIA_TEST_GDN_REFERENCE").context("set LIA_TEST_GDN_REFERENCE")?,
        );
        let qk = floats(&directory.join("qk.bin"))?;
        let conv = floats(&directory.join("conv.bin"))?;
        let decay = floats(&directory.join("decay.bin"))?;
        let beta = floats(&directory.join("beta.bin"))?;
        let expected_output = doubles(&directory.join("output.bin"))?;
        let expected_state = doubles(&directory.join("state.bin"))?;
        let batch = decay.len();
        ensure!(
            batch > 17
                && qk.len() == batch * 256
                && conv.len() == batch * 384
                && beta.len() == batch,
            "invalid single-head reference shape"
        );
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        for split in [batch, batch / 2 + 1] {
            let state = DeviceBuffer::<f32>::zeroed(&stream, 128 * 128)?;
            let prepared = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
            let mut outputs = Vec::new();
            let mut offset = 0;
            while offset < batch {
                let end = (offset + split).min(batch);
                let q = DeviceBuffer::from_host(&stream, &qk[offset * 256..end * 256])?;
                let v = DeviceBuffer::from_host(&stream, &conv[offset * 384..end * 384])?;
                let a = DeviceBuffer::from_host(&stream, &decay[offset..end])?;
                let b = DeviceBuffer::from_host(&stream, &beta[offset..end])?;
                let output = DeviceBuffer::<f32>::zeroed(&stream, (end - offset) * 128)?;
                // SAFETY: complete single-head rows, four value-shard CTAs,
                // disjoint output and persistent state, with unused preparation.
                unsafe {
                    module.delta_chunk.delta_chunk::<false>(
                        &stream,
                        LaunchConfig {
                            grid_dim: (4, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        1,
                        1,
                        &q,
                        &v,
                        &a,
                        &b,
                        &prepared,
                        state.cu_deviceptr() as *mut f32,
                        output.cu_deviceptr() as *mut f32,
                    )?;
                }
                outputs.extend(output.to_host_vec(&stream)?);
                offset = end;
            }
            compare(
                &outputs,
                &expected_output,
                &format!("outputs split={split}"),
            )?;
            compare(
                &state.to_host_vec(&stream)?,
                &expected_state,
                &format!("state split={split}"),
            )?;
        }
        Ok(())
    }
}
