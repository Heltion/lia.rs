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
        let tid = thread::index_1d().get();
        let warp = tid / 32;
        let lane = tid % 32;
        let group = warp / parts;
        let part = warp % parts;
        if group < q.len() / DIM {
            let head = group % q_heads;
            let query = group / q_heads;
            let batch = q.len() / (q_heads * DIM);
            let kh = head / (q_heads / kv_heads);
            let valid = tokens - batch + query + 1;
            let q0 = if lane < DIM {
                q[group * DIM + lane]
            } else {
                0.0
            };
            let mut o0 = 0.0f32;
            let q1 = if lane + 32 < DIM {
                q[group * DIM + lane + 32]
            } else {
                0.0
            };
            let mut o1 = 0.0f32;
            let q2 = if lane + 2 * 32 < DIM {
                q[group * DIM + lane + 2 * 32]
            } else {
                0.0
            };
            let mut o2 = 0.0f32;
            let q3 = if lane + 3 * 32 < DIM {
                q[group * DIM + lane + 3 * 32]
            } else {
                0.0
            };
            let mut o3 = 0.0f32;
            let q4 = if lane + 4 * 32 < DIM {
                q[group * DIM + lane + 4 * 32]
            } else {
                0.0
            };
            let mut o4 = 0.0f32;
            let q5 = if lane + 5 * 32 < DIM {
                q[group * DIM + lane + 5 * 32]
            } else {
                0.0
            };
            let mut o5 = 0.0f32;
            let q6 = if lane + 6 * 32 < DIM {
                q[group * DIM + lane + 6 * 32]
            } else {
                0.0
            };
            let mut o6 = 0.0f32;
            let q7 = if lane + 7 * 32 < DIM {
                q[group * DIM + lane + 7 * 32]
            } else {
                0.0
            };
            let mut o7 = 0.0f32;
            let mut max = f32::NEG_INFINITY;
            let mut sum = 0.0f32;
            let mut t = part * TILE;
            while t < (part + 1) * TILE && t < valid {
                let base = t * kv_heads * DIM + kh * DIM;
                let mut score = 0.0f32;
                if lane < DIM {
                    score = q0.mul_add(crate::quant::half(k[base + lane]), score);
                }
                if lane + 32 < DIM {
                    score = q1.mul_add(crate::quant::half(k[base + lane + 32]), score);
                }
                if lane + 2 * 32 < DIM {
                    score = q2.mul_add(crate::quant::half(k[base + lane + 2 * 32]), score);
                }
                if lane + 3 * 32 < DIM {
                    score = q3.mul_add(crate::quant::half(k[base + lane + 3 * 32]), score);
                }
                if lane + 4 * 32 < DIM {
                    score = q4.mul_add(crate::quant::half(k[base + lane + 4 * 32]), score);
                }
                if lane + 5 * 32 < DIM {
                    score = q5.mul_add(crate::quant::half(k[base + lane + 5 * 32]), score);
                }
                if lane + 6 * 32 < DIM {
                    score = q6.mul_add(crate::quant::half(k[base + lane + 6 * 32]), score);
                }
                if lane + 7 * 32 < DIM {
                    score = q7.mul_add(crate::quant::half(k[base + lane + 7 * 32]), score);
                }
                let mut delta = 16;
                while delta > 0 {
                    score += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, score, delta);
                    delta /= 2;
                }
                score = cuda_device::warp::shuffle_f32_sync(0xffff_ffff, score, 0)
                    / (DIM as f32).sqrt();
                if score <= max {
                    let beta = softmax_exp(score - max);
                    sum += beta;
                    if lane < DIM {
                        o0 = (crate::quant::half(v[base + lane])).mul_add(beta, o0);
                    }
                    if lane + 32 < DIM {
                        o1 = (crate::quant::half(v[base + lane + 32])).mul_add(beta, o1);
                    }
                    if lane + 64 < DIM {
                        o2 = (crate::quant::half(v[base + lane + 64])).mul_add(beta, o2);
                    }
                    if lane + 96 < DIM {
                        o3 = (crate::quant::half(v[base + lane + 96])).mul_add(beta, o3);
                    }
                    if lane + 128 < DIM {
                        o4 = (crate::quant::half(v[base + lane + 128])).mul_add(beta, o4);
                    }
                    if lane + 160 < DIM {
                        o5 = (crate::quant::half(v[base + lane + 160])).mul_add(beta, o5);
                    }
                    if lane + 192 < DIM {
                        o6 = (crate::quant::half(v[base + lane + 192])).mul_add(beta, o6);
                    }
                    if lane + 224 < DIM {
                        o7 = (crate::quant::half(v[base + lane + 224])).mul_add(beta, o7);
                    }
                } else {
                    let alpha = softmax_exp(max - score);
                    sum = sum.mul_add(alpha, 1.0);
                    max = score;
                    if lane < DIM {
                        o0 = o0.mul_add(alpha, crate::quant::half(v[base + lane]));
                    }
                    if lane + 32 < DIM {
                        o1 = o1.mul_add(alpha, crate::quant::half(v[base + lane + 32]));
                    }
                    if lane + 64 < DIM {
                        o2 = o2.mul_add(alpha, crate::quant::half(v[base + lane + 64]));
                    }
                    if lane + 96 < DIM {
                        o3 = o3.mul_add(alpha, crate::quant::half(v[base + lane + 96]));
                    }
                    if lane + 128 < DIM {
                        o4 = o4.mul_add(alpha, crate::quant::half(v[base + lane + 128]));
                    }
                    if lane + 160 < DIM {
                        o5 = o5.mul_add(alpha, crate::quant::half(v[base + lane + 160]));
                    }
                    if lane + 192 < DIM {
                        o6 = o6.mul_add(alpha, crate::quant::half(v[base + lane + 192]));
                    }
                    if lane + 224 < DIM {
                        o7 = o7.mul_add(alpha, crate::quant::half(v[base + lane + 224]));
                    }
                }
                t += 1;
            }
            let dest = (group * cap_parts + part) * (DIM + 2);
            unsafe {
                if lane < DIM {
                    *partial.add(dest + lane) = o0;
                }
                if lane + 32 < DIM {
                    *partial.add(dest + lane + 32) = o1;
                }
                if lane + 2 * 32 < DIM {
                    *partial.add(dest + lane + 2 * 32) = o2;
                }
                if lane + 3 * 32 < DIM {
                    *partial.add(dest + lane + 3 * 32) = o3;
                }
                if lane + 4 * 32 < DIM {
                    *partial.add(dest + lane + 4 * 32) = o4;
                }
                if lane + 5 * 32 < DIM {
                    *partial.add(dest + lane + 5 * 32) = o5;
                }
                if lane + 6 * 32 < DIM {
                    *partial.add(dest + lane + 6 * 32) = o6;
                }
                if lane + 7 * 32 < DIM {
                    *partial.add(dest + lane + 7 * 32) = o7;
                }
                if lane == 0 {
                    *partial.add(dest + DIM) = max;
                    *partial.add(dest + DIM + 1) = sum;
                }
            }
        }
    }
}
