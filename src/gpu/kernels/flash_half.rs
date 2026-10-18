use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;
    // Softmax differences are nonpositive. CUDA's hardware exponential has
    // sufficient precision here; keep recurrence gates on the standard path.
    #[inline(always)]
    fn softmax_exp(x: f32) -> f32 {
        cuda_device::float::ex2_approx_f32(x * core::f32::consts::LOG2_E)
    }

    // The host validates grouped heads, Q/K/V extents, context <= 16384,
    // batch <= 512, and padded partial storage. All indices fit u32; guarded
    // query groups and the causal token bound keep every unchecked load valid.
    #[kernel]
    pub unsafe fn flash_partial_half<const DIM: usize, const TILE: usize>(
        q_heads: usize,
        kv_heads: usize,
        tokens: usize,
        parts: usize,
        cap_parts: usize,
        q: &[f32],
        k: &[u16],
        v: &[u16],
        partial: *mut f32,
    ) {
        let tid = thread::index_1d().get() as u32;
        let q_heads = q_heads as u32;
        let kv_heads = kv_heads as u32;
        let tokens = tokens as u32;
        let parts = parts as u32;
        let cap_parts = cap_parts as u32;
        let dim = DIM as u32;
        let tile_size = TILE as u32;
        let warp = tid / 32;
        let lane = tid % 32;
        let group = warp / parts;
        let part = warp % parts;
        if group < (q.len() as u32) / dim {
            let head = group % q_heads;
            let query = group / q_heads;
            let batch = (q.len() as u32) / (q_heads * dim);
            let kh = head / (q_heads / kv_heads);
            let valid = tokens - batch + query + 1;
            let q0 = if lane < dim {
                unsafe { *q.get_unchecked((group * dim + lane) as usize) }
            } else {
                0.0
            };
            let mut o0 = 0.0f32;
            let q1 = if lane + 32 < dim {
                unsafe { *q.get_unchecked((group * dim + lane + 32) as usize) }
            } else {
                0.0
            };
            let mut o1 = 0.0f32;
            let q2 = if lane + 2 * 32 < dim {
                unsafe { *q.get_unchecked((group * dim + lane + 2 * 32) as usize) }
            } else {
                0.0
            };
            let mut o2 = 0.0f32;
            let q3 = if lane + 3 * 32 < dim {
                unsafe { *q.get_unchecked((group * dim + lane + 3 * 32) as usize) }
            } else {
                0.0
            };
            let mut o3 = 0.0f32;
            let q4 = if lane + 4 * 32 < dim {
                unsafe { *q.get_unchecked((group * dim + lane + 4 * 32) as usize) }
            } else {
                0.0
            };
            let mut o4 = 0.0f32;
            let q5 = if lane + 5 * 32 < dim {
                unsafe { *q.get_unchecked((group * dim + lane + 5 * 32) as usize) }
            } else {
                0.0
            };
            let mut o5 = 0.0f32;
            let q6 = if lane + 6 * 32 < dim {
                unsafe { *q.get_unchecked((group * dim + lane + 6 * 32) as usize) }
            } else {
                0.0
            };
            let mut o6 = 0.0f32;
            let q7 = if lane + 7 * 32 < dim {
                unsafe { *q.get_unchecked((group * dim + lane + 7 * 32) as usize) }
            } else {
                0.0
            };
            let mut o7 = 0.0f32;
            let mut max = f32::NEG_INFINITY;
            let mut sum = 0.0f32;
            let mut t = part * tile_size;
            while t < (part + 1) * tile_size && t < valid {
                let base = t * kv_heads * dim + kh * dim;
                let mut score = 0.0f32;
                if lane < dim {
                    score = q0.mul_add(
                        crate::quant::half(unsafe { *k.get_unchecked((base + lane) as usize) }),
                        score,
                    );
                }
                if lane + 32 < dim {
                    score = q1.mul_add(
                        crate::quant::half(unsafe {
                            *k.get_unchecked((base + lane + 32) as usize)
                        }),
                        score,
                    );
                }
                if lane + 2 * 32 < dim {
                    score = q2.mul_add(
                        crate::quant::half(unsafe {
                            *k.get_unchecked((base + lane + 2 * 32) as usize)
                        }),
                        score,
                    );
                }
                if lane + 3 * 32 < dim {
                    score = q3.mul_add(
                        crate::quant::half(unsafe {
                            *k.get_unchecked((base + lane + 3 * 32) as usize)
                        }),
                        score,
                    );
                }
                if lane + 4 * 32 < dim {
                    score = q4.mul_add(
                        crate::quant::half(unsafe {
                            *k.get_unchecked((base + lane + 4 * 32) as usize)
                        }),
                        score,
                    );
                }
                if lane + 5 * 32 < dim {
                    score = q5.mul_add(
                        crate::quant::half(unsafe {
                            *k.get_unchecked((base + lane + 5 * 32) as usize)
                        }),
                        score,
                    );
                }
                if lane + 6 * 32 < dim {
                    score = q6.mul_add(
                        crate::quant::half(unsafe {
                            *k.get_unchecked((base + lane + 6 * 32) as usize)
                        }),
                        score,
                    );
                }
                if lane + 7 * 32 < dim {
                    score = q7.mul_add(
                        crate::quant::half(unsafe {
                            *k.get_unchecked((base + lane + 7 * 32) as usize)
                        }),
                        score,
                    );
                }
                let mut delta = 16;
                while delta > 0 {
                    score += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, score, delta);
                    delta /= 2;
                }
                score = cuda_device::warp::shuffle_f32_sync(0xffff_ffff, score, 0)
                    / (dim as f32).sqrt();
                if score <= max {
                    let beta = softmax_exp(score - max);
                    sum += beta;
                    if lane < dim {
                        o0 = (crate::quant::half(unsafe {
                            *v.get_unchecked((base + lane) as usize)
                        }))
                        .mul_add(beta, o0);
                    }
                    if lane + 32 < dim {
                        o1 = (crate::quant::half(unsafe {
                            *v.get_unchecked((base + lane + 32) as usize)
                        }))
                        .mul_add(beta, o1);
                    }
                    if lane + 64 < dim {
                        o2 = (crate::quant::half(unsafe {
                            *v.get_unchecked((base + lane + 64) as usize)
                        }))
                        .mul_add(beta, o2);
                    }
                    if lane + 96 < dim {
                        o3 = (crate::quant::half(unsafe {
                            *v.get_unchecked((base + lane + 96) as usize)
                        }))
                        .mul_add(beta, o3);
                    }
                    if lane + 128 < dim {
                        o4 = (crate::quant::half(unsafe {
                            *v.get_unchecked((base + lane + 128) as usize)
                        }))
                        .mul_add(beta, o4);
                    }
                    if lane + 160 < dim {
                        o5 = (crate::quant::half(unsafe {
                            *v.get_unchecked((base + lane + 160) as usize)
                        }))
                        .mul_add(beta, o5);
                    }
                    if lane + 192 < dim {
                        o6 = (crate::quant::half(unsafe {
                            *v.get_unchecked((base + lane + 192) as usize)
                        }))
                        .mul_add(beta, o6);
                    }
                    if lane + 224 < dim {
                        o7 = (crate::quant::half(unsafe {
                            *v.get_unchecked((base + lane + 224) as usize)
                        }))
                        .mul_add(beta, o7);
                    }
                } else {
                    let alpha = softmax_exp(max - score);
                    sum = sum.mul_add(alpha, 1.0);
                    max = score;
                    if lane < dim {
                        o0 = o0.mul_add(
                            alpha,
                            crate::quant::half(unsafe { *v.get_unchecked((base + lane) as usize) }),
                        );
                    }
                    if lane + 32 < dim {
                        o1 = o1.mul_add(
                            alpha,
                            crate::quant::half(unsafe {
                                *v.get_unchecked((base + lane + 32) as usize)
                            }),
                        );
                    }
                    if lane + 64 < dim {
                        o2 = o2.mul_add(
                            alpha,
                            crate::quant::half(unsafe {
                                *v.get_unchecked((base + lane + 64) as usize)
                            }),
                        );
                    }
                    if lane + 96 < dim {
                        o3 = o3.mul_add(
                            alpha,
                            crate::quant::half(unsafe {
                                *v.get_unchecked((base + lane + 96) as usize)
                            }),
                        );
                    }
                    if lane + 128 < dim {
                        o4 = o4.mul_add(
                            alpha,
                            crate::quant::half(unsafe {
                                *v.get_unchecked((base + lane + 128) as usize)
                            }),
                        );
                    }
                    if lane + 160 < dim {
                        o5 = o5.mul_add(
                            alpha,
                            crate::quant::half(unsafe {
                                *v.get_unchecked((base + lane + 160) as usize)
                            }),
                        );
                    }
                    if lane + 192 < dim {
                        o6 = o6.mul_add(
                            alpha,
                            crate::quant::half(unsafe {
                                *v.get_unchecked((base + lane + 192) as usize)
                            }),
                        );
                    }
                    if lane + 224 < dim {
                        o7 = o7.mul_add(
                            alpha,
                            crate::quant::half(unsafe {
                                *v.get_unchecked((base + lane + 224) as usize)
                            }),
                        );
                    }
                }
                t += 1;
            }
            let dest = (group * cap_parts + part) * (dim + 2);
            unsafe {
                if lane < dim {
                    *partial.add((dest + lane) as usize) = o0;
                }
                if lane + 32 < dim {
                    *partial.add((dest + lane + 32) as usize) = o1;
                }
                if lane + 2 * 32 < dim {
                    *partial.add((dest + lane + 2 * 32) as usize) = o2;
                }
                if lane + 3 * 32 < dim {
                    *partial.add((dest + lane + 3 * 32) as usize) = o3;
                }
                if lane + 4 * 32 < dim {
                    *partial.add((dest + lane + 4 * 32) as usize) = o4;
                }
                if lane + 5 * 32 < dim {
                    *partial.add((dest + lane + 5 * 32) as usize) = o5;
                }
                if lane + 6 * 32 < dim {
                    *partial.add((dest + lane + 6 * 32) as usize) = o6;
                }
                if lane + 7 * 32 < dim {
                    *partial.add((dest + lane + 7 * 32) as usize) = o7;
                }
                if lane == 0 {
                    *partial.add((dest + dim) as usize) = max;
                    *partial.add((dest + dim + 1) as usize) = sum;
                }
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
    fn partitioned_attention() -> Result<()> {
        check::<64>(65, 2, 4, 2)?;
        check::<128>(129, 2, 4, 2)?;
        check::<64>(513, 17, 4, 2)?;
        check::<128>(513, 33, 4, 2)?;
        check::<64>(2049, 1, 24, 4)?;
        check::<64>(2049, 1, 16, 2)?;
        check::<128>(513, 3, 16, 2)?;
        check::<64>(2049, 4, 24, 4)?;
        check::<128>(513, 17, 24, 4)?;
        check::<128>(1025, 129, 4, 2)?;
        check::<128>(33, 33, 4, 2)
    }

    fn check<const TILE: usize>(tokens: usize, batch: usize, qh: usize, kvh: usize) -> Result<()> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let module = kernels::load(&ctx)?;
        let dim = 256;
        let parts = tokens.div_ceil(TILE);
        let cap = parts + 1;
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
        // SAFETY: the causal prefix and both queries fit every cache; the
        // partial allocation includes padding beyond the active partitions.
        unsafe {
            module.flash.flash_partial::<256, TILE>(
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
            module.attention.flash_merge::<256>(
                &stream,
                LaunchConfig::for_num_elems((batch * qh * 32) as u32),
                batch * qh,
                parts,
                cap,
                0,
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
            "GPU 256-dimensional online attention matches f64 across context partitions and causal query tails"
        );
        let k16 = DeviceBuffer::<u16>::zeroed(&stream, k.len())?;
        let v16 = DeviceBuffer::<u16>::zeroed(&stream, v.len())?;
        // SAFETY: exact cache extents, full causal context and one producer per cache element.
        unsafe {
            module.rotary.cache_kv_half(
                &stream,
                LaunchConfig::for_num_elems(k.len() as u32),
                kvh * dim,
                0,
                &dk,
                &dv,
                k16.cu_deviceptr() as *mut u16,
                v16.cu_deviceptr() as *mut u16,
            )?;
            module.flash_half.flash_partial_half::<256, TILE>(
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
            module.attention.flash_merge::<256>(
                &stream,
                LaunchConfig::for_num_elems((batch * qh * 32) as u32),
                batch * qh,
                parts,
                cap,
                0,
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
        for span in [256, 512, 1024, tokens.div_ceil(32) * 32] {
            let mma_parts = tokens.div_ceil(span);
            for grouped in [false, true] {
                let mut baseline = None;
                for stride in [batch, 16] {
                    let mma_partial = DeviceBuffer::<f32>::zeroed(
                        &stream,
                        batch.min(stride) * qh * mma_parts * 258,
                    )?;
                    for query_start in (0..batch).step_by(stride) {
                        let query_count = (batch - query_start).min(stride);
                        let heads = if grouped { kvh } else { qh };
                        let queries = query_count * if grouped { qh / kvh } else { 1 };
                        let config = LaunchConfig {
                            grid_dim: ((queries.div_ceil(16) * heads * mma_parts) as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        // SAFETY: independent complete causal queries, matching GQA grouping and disjoint output slots.
                        unsafe {
                            if grouped {
                                module.flash_mma.flash_mma::<true>(
                                    &stream,
                                    config,
                                    qh,
                                    kvh,
                                    tokens,
                                    span,
                                    mma_parts,
                                    query_start,
                                    &dq,
                                    &k16,
                                    &v16,
                                    mma_partial.cu_deviceptr() as *mut f32,
                                )?;
                            } else {
                                module.flash_mma.flash_mma::<false>(
                                    &stream,
                                    config,
                                    qh,
                                    kvh,
                                    tokens,
                                    span,
                                    mma_parts,
                                    query_start,
                                    &dq,
                                    &k16,
                                    &v16,
                                    mma_partial.cu_deviceptr() as *mut f32,
                                )?;
                            }
                            module.attention.flash_merge::<256>(
                                &stream,
                                LaunchConfig::for_num_elems((query_count * qh * 32) as u32),
                                query_count * qh,
                                mma_parts,
                                mma_parts,
                                query_start * qh,
                                &mma_partial,
                                &dg,
                                out.cu_deviceptr() as *mut f32,
                            )?;
                        }
                    }
                    let actual = out.to_host_vec(&stream)?;
                    if let Some(previous) = &baseline {
                        assert_eq!(
                            &actual, previous,
                            "query tiles change attention: batch={batch}, grouped={grouped}, span={span}"
                        );
                    } else {
                        baseline = Some(actual.clone());
                    }
                    close(&actual, &expected, "Tensor Core grouped causal attention")?;
                }
            }
        }
        Ok(())
    }
}
