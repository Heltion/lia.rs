//! Rust kernels compiled to PTX by NVIDIA cuda-oxide.
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, simt::LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

// Kernel ABIs keep dimensions and device buffers explicit for launch auditing.
#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod kernels {
    use super::*;

    // Softmax differences are nonpositive. CUDA's hardware exponential has
    // sufficient precision here; keep recurrence gates on the standard path.
    #[inline(always)]
    fn softmax_exp(x: f32) -> f32 {
        cuda_device::float::ex2_approx_f32(x * core::f32::consts::LOG2_E)
    }

    /// Each warp reduces a disjoint vocabulary partition. Keys use Rust's
    /// total float order; ties choose the higher token index like Iterator::max.
    #[kernel]
    pub unsafe fn argmax_partial(
        vocab: usize,
        parts: usize,
        input: &[f32],
        values: *mut f32,
        ids: *mut u32,
    ) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        if group < input.len() / vocab * parts {
            let token = group / parts;
            let part = group % parts;
            let mut best = f32::NEG_INFINITY;
            let mut best_key = i32::MIN;
            let mut best_id = 0u32;
            let mut i = part * 1024 + lane;
            while i < (part + 1) * 1024 && i < vocab {
                let value = input[token * vocab + i];
                let bits = value.to_bits() as i32;
                let key = bits ^ (((bits >> 31) as u32 >> 1) as i32);
                if key > best_key || (key == best_key && i as u32 > best_id) {
                    best = value;
                    best_key = key;
                    best_id = i as u32;
                }
                i += 32;
            }
            let mut delta = 16;
            while delta > 0 {
                let other = cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, best, delta);
                let other_id = cuda_device::warp::shuffle_down_f32_sync(
                    0xffff_ffff,
                    f32::from_bits(best_id),
                    delta,
                )
                .to_bits();
                let bits = other.to_bits() as i32;
                let key = bits ^ (((bits >> 31) as u32 >> 1) as i32);
                if key > best_key || (key == best_key && other_id > best_id) {
                    best = other;
                    best_key = key;
                    best_id = other_id;
                }
                delta /= 2;
            }
            if lane == 0 {
                unsafe {
                    *values.add(group) = best;
                    *ids.add(group) = best_id;
                }
            }
        }
    }

    /// Copy one already validated position from a speculative state history.
    #[kernel]
    pub unsafe fn restore_position(offset: usize, count: usize, input: &[f32], out: *mut f32) {
        let i = thread::index_1d().get();
        if i < count {
            unsafe {
                *out.add(i) = input[offset + i];
            }
        }
    }

    #[kernel]
    pub fn concat(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = out.get_mut(idx) {
            *value = if i < a.len() { a[i] } else { b[i - a.len()] };
        }
    }

    #[kernel]
    pub fn embedding(
        kind: u32,
        bytes: usize,
        block_elems: usize,
        dim: usize,
        token: usize,
        weights: &[u8],
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = out.get_mut(idx) {
            let block = (token * dim + i) / block_elems;
            *value = crate::quant::decode_value(
                kind,
                &weights[block * bytes..(block + 1) * bytes],
                i % block_elems,
            );
        }
    }

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

    #[kernel]
    pub unsafe fn delta_net<const PRECOMPUTED: bool>(
        dim: usize,
        k_heads: usize,
        v_heads: usize,
        qk: &[f32],
        conv: &[f32],
        alpha: &[f32],
        beta: &[f32],
        dt: &[f32],
        a: &[f32],
        state: *mut f32,
        captured: *mut f32,
        out: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let row = tid / 32;
        let lane = tid % 32;
        let head = row / dim;
        let value = row % dim;
        if head < v_heads {
            let mut t = 0;
            while t < alpha.len() / v_heads {
                let kh = head % k_heads;
                let qoff = t * 2 * k_heads * dim + kh * dim;
                let koff = t * 2 * k_heads * dim + (k_heads + kh) * dim;
                let v = conv
                    [t * (2 * k_heads + v_heads) * dim + 2 * k_heads * dim + head * dim + value];
                let b = if PRECOMPUTED {
                    beta[t * v_heads + head]
                } else {
                    1.0 / (1.0 + (-beta[t * v_heads + head]).exp())
                };
                let decay = if PRECOMPUTED {
                    alpha[t * v_heads + head]
                } else {
                    let x = alpha[t * v_heads + head] + dt[head];
                    let softplus = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
                    (a[head] * softplus).exp()
                };
                let mut dot = 0.0f32;
                let mut j = lane;
                while j < dim {
                    unsafe {
                        dot += *state.add(row * dim + j) * qk[koff + j];
                    }
                    j += 32;
                }
                let mut delta = 16;
                while delta > 0 {
                    dot += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, dot, delta);
                    delta /= 2;
                }
                dot = cuda_device::warp::shuffle_f32_sync(0xffff_ffff, dot, 0);
                let correction = (v - decay * dot) * b;
                let mut result = 0.0f32;
                j = lane;
                while j < dim {
                    unsafe {
                        let s = decay * *state.add(row * dim + j) + qk[koff + j] * correction;
                        *state.add(row * dim + j) = s;
                        if !captured.is_null() {
                            *captured.add(t * v_heads * dim * dim + row * dim + j) = s;
                        }
                        result += s * qk[qoff + j];
                    }
                    j += 32;
                }
                delta = 16;
                while delta > 0 {
                    result += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, result, delta);
                    delta /= 2;
                }
                if lane == 0 {
                    unsafe {
                        *out.add(t * v_heads * dim + row) = result / (dim as f32).sqrt();
                    }
                }
                t += 1;
            }
        }
    }

    /// A 128-wide recurrent row has four elements per lane. Keep them live
    /// across the token batch instead of rereading the matrix for every step.
    #[kernel]
    pub unsafe fn delta_net_128(
        k_heads: usize,
        v_heads: usize,
        qk: &[f32],
        conv: &[f32],
        decay: &[f32],
        beta: &[f32],
        state: *mut f32,
        captured: *mut f32,
        out: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let row = tid / 32;
        let lane = tid % 32;
        let head = row / 128;
        let value = row % 128;
        if head < v_heads {
            let base = row * 128 + lane;
            let (mut s0, mut s1, mut s2, mut s3) = unsafe {
                (
                    *state.add(base),
                    *state.add(base + 32),
                    *state.add(base + 64),
                    *state.add(base + 96),
                )
            };
            let mut t = 0;
            while t < decay.len() / v_heads {
                let qoff = t * 2 * k_heads * 128 + (head % k_heads) * 128 + lane;
                let koff = qoff + k_heads * 128;
                let k0 = qk[koff];
                let k1 = qk[koff + 32];
                let k2 = qk[koff + 64];
                let k3 = qk[koff + 96];
                let mut dot = s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3;
                let mut delta = 16;
                while delta > 0 {
                    dot += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, dot, delta);
                    delta /= 2;
                }
                dot = cuda_device::warp::shuffle_f32_sync(0xffff_ffff, dot, 0);
                let d = decay[t * v_heads + head];
                let v = conv
                    [t * (2 * k_heads + v_heads) * 128 + 2 * k_heads * 128 + head * 128 + value];
                let correction = (v - d * dot) * beta[t * v_heads + head];
                s0 = d * s0 + k0 * correction;
                s1 = d * s1 + k1 * correction;
                s2 = d * s2 + k2 * correction;
                s3 = d * s3 + k3 * correction;
                if !captured.is_null() {
                    unsafe {
                        let off = t * v_heads * 128 * 128 + base;
                        *captured.add(off) = s0;
                        *captured.add(off + 32) = s1;
                        *captured.add(off + 64) = s2;
                        *captured.add(off + 96) = s3;
                    }
                }
                let mut result =
                    s0 * qk[qoff] + s1 * qk[qoff + 32] + s2 * qk[qoff + 64] + s3 * qk[qoff + 96];
                delta = 16;
                while delta > 0 {
                    result += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, result, delta);
                    delta /= 2;
                }
                if lane == 0 {
                    unsafe {
                        *out.add(t * v_heads * 128 + row) = result / 128.0f32.sqrt();
                    }
                }
                t += 1;
            }
            unsafe {
                *state.add(base) = s0;
                *state.add(base + 32) = s1;
                *state.add(base + 64) = s2;
                *state.add(base + 96) = s3;
            }
        }
    }

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
    pub unsafe fn flash_partial<const DIM: usize, const TILE: usize>(
        q_heads: usize,
        kv_heads: usize,
        tokens: usize,
        parts: usize,
        cap_parts: usize,
        q: &[f32],
        k: &[f32],
        v: &[f32],
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
                    score = q0.mul_add(k[base + lane], score);
                }
                if lane + 32 < DIM {
                    score = q1.mul_add(k[base + lane + 32], score);
                }
                if lane + 2 * 32 < DIM {
                    score = q2.mul_add(k[base + lane + 2 * 32], score);
                }
                if lane + 3 * 32 < DIM {
                    score = q3.mul_add(k[base + lane + 3 * 32], score);
                }
                if lane + 4 * 32 < DIM {
                    score = q4.mul_add(k[base + lane + 4 * 32], score);
                }
                if lane + 5 * 32 < DIM {
                    score = q5.mul_add(k[base + lane + 5 * 32], score);
                }
                if lane + 6 * 32 < DIM {
                    score = q6.mul_add(k[base + lane + 6 * 32], score);
                }
                if lane + 7 * 32 < DIM {
                    score = q7.mul_add(k[base + lane + 7 * 32], score);
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
                        o0 = (v[base + lane]).mul_add(beta, o0);
                    }
                    if lane + 32 < DIM {
                        o1 = (v[base + lane + 32]).mul_add(beta, o1);
                    }
                    if lane + 64 < DIM {
                        o2 = (v[base + lane + 64]).mul_add(beta, o2);
                    }
                    if lane + 96 < DIM {
                        o3 = (v[base + lane + 96]).mul_add(beta, o3);
                    }
                    if lane + 128 < DIM {
                        o4 = (v[base + lane + 128]).mul_add(beta, o4);
                    }
                    if lane + 160 < DIM {
                        o5 = (v[base + lane + 160]).mul_add(beta, o5);
                    }
                    if lane + 192 < DIM {
                        o6 = (v[base + lane + 192]).mul_add(beta, o6);
                    }
                    if lane + 224 < DIM {
                        o7 = (v[base + lane + 224]).mul_add(beta, o7);
                    }
                } else {
                    let alpha = softmax_exp(max - score);
                    sum = sum.mul_add(alpha, 1.0);
                    max = score;
                    if lane < DIM {
                        o0 = o0.mul_add(alpha, v[base + lane]);
                    }
                    if lane + 32 < DIM {
                        o1 = o1.mul_add(alpha, v[base + lane + 32]);
                    }
                    if lane + 64 < DIM {
                        o2 = o2.mul_add(alpha, v[base + lane + 64]);
                    }
                    if lane + 96 < DIM {
                        o3 = o3.mul_add(alpha, v[base + lane + 96]);
                    }
                    if lane + 128 < DIM {
                        o4 = o4.mul_add(alpha, v[base + lane + 128]);
                    }
                    if lane + 160 < DIM {
                        o5 = o5.mul_add(alpha, v[base + lane + 160]);
                    }
                    if lane + 192 < DIM {
                        o6 = o6.mul_add(alpha, v[base + lane + 192]);
                    }
                    if lane + 224 < DIM {
                        o7 = o7.mul_add(alpha, v[base + lane + 224]);
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

    #[kernel]
    pub unsafe fn flash_merge<const DIM: usize>(
        groups: usize,
        parts: usize,
        cap_parts: usize,
        partial: &[f32],
        qgate: &[f32],
        out: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        if group < groups {
            let mut max = f32::NEG_INFINITY;
            let mut p = 0;
            while p < parts {
                max = max.max(partial[(group * cap_parts + p) * (DIM + 2) + DIM]);
                p += 1;
            }
            let mut sum = 0.0f32;
            let mut o0 = 0.0f32;
            let mut o1 = 0.0f32;
            let mut o2 = 0.0f32;
            let mut o3 = 0.0f32;
            let mut o4 = 0.0f32;
            let mut o5 = 0.0f32;
            let mut o6 = 0.0f32;
            let mut o7 = 0.0f32;
            p = 0;
            while p < parts {
                let base = (group * cap_parts + p) * (DIM + 2);
                let weight = softmax_exp(partial[base + DIM] - max);
                sum += weight * partial[base + DIM + 1];
                if lane < DIM {
                    o0 += weight * partial[base + lane];
                }
                if lane + 32 < DIM {
                    o1 += weight * partial[base + lane + 32];
                }
                if lane + 2 * 32 < DIM {
                    o2 += weight * partial[base + lane + 2 * 32];
                }
                if lane + 3 * 32 < DIM {
                    o3 += weight * partial[base + lane + 3 * 32];
                }
                if lane + 4 * 32 < DIM {
                    o4 += weight * partial[base + lane + 4 * 32];
                }
                if lane + 5 * 32 < DIM {
                    o5 += weight * partial[base + lane + 5 * 32];
                }
                if lane + 6 * 32 < DIM {
                    o6 += weight * partial[base + lane + 6 * 32];
                }
                if lane + 7 * 32 < DIM {
                    o7 += weight * partial[base + lane + 7 * 32];
                }
                p += 1;
            }
            unsafe {
                if lane < DIM {
                    *out.add(group * DIM + lane) =
                        o0 / sum / (1.0 + (-qgate[group * DIM * 2 + DIM + lane]).exp());
                }
                if lane + 32 < DIM {
                    *out.add(group * DIM + lane + 32) =
                        o1 / sum / (1.0 + (-qgate[group * DIM * 2 + DIM + lane + 32]).exp());
                }
                if lane + 2 * 32 < DIM {
                    *out.add(group * DIM + lane + 2 * 32) =
                        o2 / sum / (1.0 + (-qgate[group * DIM * 2 + DIM + lane + 2 * 32]).exp());
                }
                if lane + 3 * 32 < DIM {
                    *out.add(group * DIM + lane + 3 * 32) =
                        o3 / sum / (1.0 + (-qgate[group * DIM * 2 + DIM + lane + 3 * 32]).exp());
                }
                if lane + 4 * 32 < DIM {
                    *out.add(group * DIM + lane + 4 * 32) =
                        o4 / sum / (1.0 + (-qgate[group * DIM * 2 + DIM + lane + 4 * 32]).exp());
                }
                if lane + 5 * 32 < DIM {
                    *out.add(group * DIM + lane + 5 * 32) =
                        o5 / sum / (1.0 + (-qgate[group * DIM * 2 + DIM + lane + 5 * 32]).exp());
                }
                if lane + 6 * 32 < DIM {
                    *out.add(group * DIM + lane + 6 * 32) =
                        o6 / sum / (1.0 + (-qgate[group * DIM * 2 + DIM + lane + 6 * 32]).exp());
                }
                if lane + 7 * 32 < DIM {
                    *out.add(group * DIM + lane + 7 * 32) =
                        o7 / sum / (1.0 + (-qgate[group * DIM * 2 + DIM + lane + 7 * 32]).exp());
                }
            }
        }
    }

    #[kernel]
    pub unsafe fn attn_scores(
        dim: usize,
        q_heads: usize,
        kv_heads: usize,
        tokens: usize,
        capacity: usize,
        q: &[f32],
        k: &[f32],
        scores: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let row = tid / 32;
        let lane = tid % 32;
        let group = row / tokens;
        let head = group % q_heads;
        let query = group / q_heads;
        let t = row % tokens;
        let batch = q.len() / (q_heads * dim);
        if query < batch {
            if t > tokens - batch + query {
                if lane == 0 {
                    unsafe {
                        *scores.add(group * capacity + t) = f32::NEG_INFINITY;
                    }
                }
                return;
            }
            let kh = head / (q_heads / kv_heads);
            let mut sum = 0.0f32;
            let mut j = lane;
            while j < dim {
                sum += q[group * dim + j] * k[t * kv_heads * dim + kh * dim + j];
                j += 32;
            }
            let mut delta = 16;
            while delta > 0 {
                sum += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum, delta);
                delta /= 2;
            }
            if lane == 0 {
                unsafe {
                    *scores.add(group * capacity + t) = sum / (dim as f32).sqrt();
                }
            }
        }
    }

    #[kernel]
    pub unsafe fn softmax(heads: usize, tokens: usize, capacity: usize, scores: *mut f32) {
        let tid = thread::index_1d().get();
        let head = tid / 32;
        let lane = tid % 32;
        if head < heads {
            let mut max = f32::NEG_INFINITY;
            let mut j = lane;
            while j < tokens {
                unsafe {
                    max = max.max(*scores.add(head * capacity + j));
                }
                j += 32;
            }
            let mut delta = 16;
            while delta > 0 {
                max = max.max(cuda_device::warp::shuffle_down_f32_sync(
                    0xffff_ffff,
                    max,
                    delta,
                ));
                delta /= 2;
            }
            max = cuda_device::warp::shuffle_f32_sync(0xffff_ffff, max, 0);
            let mut sum = 0.0f32;
            j = lane;
            while j < tokens {
                unsafe {
                    sum += (*scores.add(head * capacity + j) - max).exp();
                }
                j += 32;
            }
            delta = 16;
            while delta > 0 {
                sum += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum, delta);
                delta /= 2;
            }
            sum = cuda_device::warp::shuffle_f32_sync(0xffff_ffff, sum, 0);
            j = lane;
            while j < tokens {
                unsafe {
                    let p = scores.add(head * capacity + j);
                    *p = (*p - max).exp() / sum;
                }
                j += 32;
            }
        }
    }

    #[kernel]
    pub fn attn_values(
        dim: usize,
        q_heads: usize,
        kv_heads: usize,
        tokens: usize,
        capacity: usize,
        probs: &[f32],
        v: &[f32],
        qgate: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = out.get_mut(idx) {
            let head = i / dim;
            let lane = i % dim;
            let kh = (head % q_heads) / (q_heads / kv_heads);
            let mut sum = 0.0f32;
            let mut t = 0;
            while t < tokens {
                sum += probs[head * capacity + t] * v[t * kv_heads * dim + kh * dim + lane];
                t += 1;
            }
            *value = sum / (1.0 + (-qgate[head * dim * 2 + dim + lane]).exp());
        }
    }

    #[kernel]
    pub unsafe fn quantize_activation(input: &[f32], packed: *mut u32, scales: *mut f32) {
        let i = thread::index_1d().get();
        if i < input.len() {
            let lane = (i % 32) as u32;
            let mut max = input[i].abs();
            let mut delta = 16;
            while delta > 0 {
                max = max.max(cuda_device::warp::shuffle_down_f32_sync(
                    0xffff_ffff,
                    max,
                    delta,
                ));
                delta /= 2;
            }
            max = cuda_device::warp::shuffle_f32_sync(0xffff_ffff, max, 0);
            let scale = if max == 0.0 { 1.0 } else { max / 127.0 };
            let q = (input[i] / scale).round_ties_even() as i32 as u32;
            let base = lane / 4 * 4;
            let q0 = cuda_device::warp::shuffle_sync(0xffff_ffff, q, base) & 255;
            let q1 = cuda_device::warp::shuffle_sync(0xffff_ffff, q, base + 1) & 255;
            let q2 = cuda_device::warp::shuffle_sync(0xffff_ffff, q, base + 2) & 255;
            let q3 = cuda_device::warp::shuffle_sync(0xffff_ffff, q, base + 3) & 255;
            unsafe {
                if lane.is_multiple_of(4) {
                    *packed.add(i / 4) = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
                }
                if lane == 0 {
                    *scales.add(i / 32) = scale;
                }
            }
        }
    }

    #[inline(always)]
    fn int_dot16<const KIND: u32>(
        w0: u32,
        w1: u32,
        w2: u32,
        w3: u32,
        q0: u32,
        q1: u32,
        q2: u32,
        q3: u32,
        scale: f32,
        offset0: f32,
        offset1: f32,
    ) -> f32 {
        let dot = cuda_device::dotprod::dp4a_s32(w0, q0, 0);
        let dot = cuda_device::dotprod::dp4a_s32(w1, q1, dot);
        let dot = cuda_device::dotprod::dp4a_s32(w2, q2, dot);
        let dot = cuda_device::dotprod::dp4a_s32(w3, q3, dot) as f32;
        if KIND == 10 || KIND == 12 || KIND == 19 || KIND == 29 {
            let a = cuda_device::dotprod::dp4a_s32(0x01010101, q0, 0);
            let a = cuda_device::dotprod::dp4a_s32(0x01010101, q1, a);
            let b = cuda_device::dotprod::dp4a_s32(0x01010101, q2, 0);
            let b = cuda_device::dotprod::dp4a_s32(0x01010101, q3, b);
            if KIND == 29 {
                scale * dot + offset0 * a as f32 + offset1 * b as f32
            } else {
                scale * dot + offset0 * (a + b) as f32
            }
        } else {
            scale * dot
        }
    }

    #[kernel]
    pub unsafe fn matmul_int<
        const KIND: u32,
        const BYTES: usize,
        const MAX: usize,
        const SPLIT: usize,
        const ROW_GROUP: usize,
    >(
        cols: usize,
        rows: usize,
        batch: usize,
        weights: &[u8],
        x: &[u32],
        scales: &[f32],
        out: *mut f32,
    ) {
        static mut PARTIAL: cuda_device::SharedArray<f32, 64> = cuda_device::SharedArray::UNINIT;
        let tid = thread::index_1d().get() as u32;
        let row = (tid / (32 * SPLIT as u32)) * ROW_GROUP as u32;
        let split = (tid / 32) % SPLIT as u32;
        let lane = tid % 32;
        let cols = cols as u32;
        let rows = rows as u32;
        let batch = batch as u32;
        if row < rows {
            let mut sum0 = 0.0f32;
            let mut sum1 = 0.0f32;
            let mut sum2 = 0.0f32;
            let mut sum3 = 0.0f32;
            let mut sum4 = 0.0f32;
            let mut sum5 = 0.0f32;
            let mut sum6 = 0.0f32;
            let mut sum7 = 0.0f32;
            let mut sum0_b = 0.0f32;
            let mut sum1_b = 0.0f32;
            let mut sum2_b = 0.0f32;
            let mut sum3_b = 0.0f32;
            let mut sum4_b = 0.0f32;
            let mut sum5_b = 0.0f32;
            let mut sum6_b = 0.0f32;
            let mut sum7_b = 0.0f32;
            let mut col = (split * 32 + lane) * 16;
            while col < cols {
                let block = (row * cols + col) / 256;
                let data = unsafe {
                    core::slice::from_raw_parts(
                        weights.as_ptr().add((block * BYTES as u32) as usize),
                        BYTES,
                    )
                };
                let i = (col & 240) as usize;
                let (w0, w1, scale, offset0) =
                    unsafe { crate::quant::packed_eight_aligned(KIND, data, i) };
                let (w2, w3, _, offset1) =
                    unsafe { crate::quant::packed_eight_aligned(KIND, data, i + 8) };
                let (v0, v1, v2, v3, scale_b, offset0_b, offset1_b) =
                    if ROW_GROUP == 2 && row + 1 < rows {
                        let block = ((row + 1) * cols + col) / 256;
                        let data = unsafe {
                            core::slice::from_raw_parts(
                                weights.as_ptr().add((block * BYTES as u32) as usize),
                                BYTES,
                            )
                        };
                        let (a, b, scale, off0) =
                            unsafe { crate::quant::packed_eight_aligned(KIND, data, i) };
                        let (c, d, _, off1) =
                            unsafe { crate::quant::packed_eight_aligned(KIND, data, i + 8) };
                        (a, b, c, d, scale, off0, off1)
                    } else {
                        (0, 0, 0, 0, 0.0, 0.0, 0.0)
                    };
                if MAX > 0 && batch > 0 {
                    let base = (col) / 4;
                    let q0 = unsafe { *x.get_unchecked(base as usize) };
                    let q1 = unsafe { *x.get_unchecked((base + 1) as usize) };
                    let q2 = unsafe { *x.get_unchecked((base + 2) as usize) };
                    let q3 = unsafe { *x.get_unchecked((base + 3) as usize) };
                    let xs = unsafe { *scales.get_unchecked(((col) / 32) as usize) };
                    sum0 += xs
                        * int_dot16::<KIND>(
                            w0, w1, w2, w3, q0, q1, q2, q3, scale, offset0, offset1,
                        );
                    if ROW_GROUP == 2 && row + 1 < rows {
                        sum0_b += xs
                            * int_dot16::<KIND>(
                                v0, v1, v2, v3, q0, q1, q2, q3, scale_b, offset0_b, offset1_b,
                            );
                    }
                }
                if MAX > 1 && batch > 1 {
                    let base = (cols + col) / 4;
                    let q0 = unsafe { *x.get_unchecked(base as usize) };
                    let q1 = unsafe { *x.get_unchecked((base + 1) as usize) };
                    let q2 = unsafe { *x.get_unchecked((base + 2) as usize) };
                    let q3 = unsafe { *x.get_unchecked((base + 3) as usize) };
                    let xs = unsafe { *scales.get_unchecked(((cols + col) / 32) as usize) };
                    sum1 += xs
                        * int_dot16::<KIND>(
                            w0, w1, w2, w3, q0, q1, q2, q3, scale, offset0, offset1,
                        );
                    if ROW_GROUP == 2 && row + 1 < rows {
                        sum1_b += xs
                            * int_dot16::<KIND>(
                                v0, v1, v2, v3, q0, q1, q2, q3, scale_b, offset0_b, offset1_b,
                            );
                    }
                }
                if MAX > 2 && batch > 2 {
                    let base = (2 * cols + col) / 4;
                    let q0 = unsafe { *x.get_unchecked(base as usize) };
                    let q1 = unsafe { *x.get_unchecked((base + 1) as usize) };
                    let q2 = unsafe { *x.get_unchecked((base + 2) as usize) };
                    let q3 = unsafe { *x.get_unchecked((base + 3) as usize) };
                    let xs = unsafe { *scales.get_unchecked(((2 * cols + col) / 32) as usize) };
                    sum2 += xs
                        * int_dot16::<KIND>(
                            w0, w1, w2, w3, q0, q1, q2, q3, scale, offset0, offset1,
                        );
                    if ROW_GROUP == 2 && row + 1 < rows {
                        sum2_b += xs
                            * int_dot16::<KIND>(
                                v0, v1, v2, v3, q0, q1, q2, q3, scale_b, offset0_b, offset1_b,
                            );
                    }
                }
                if MAX > 3 && batch > 3 {
                    let base = (3 * cols + col) / 4;
                    let q0 = unsafe { *x.get_unchecked(base as usize) };
                    let q1 = unsafe { *x.get_unchecked((base + 1) as usize) };
                    let q2 = unsafe { *x.get_unchecked((base + 2) as usize) };
                    let q3 = unsafe { *x.get_unchecked((base + 3) as usize) };
                    let xs = unsafe { *scales.get_unchecked(((3 * cols + col) / 32) as usize) };
                    sum3 += xs
                        * int_dot16::<KIND>(
                            w0, w1, w2, w3, q0, q1, q2, q3, scale, offset0, offset1,
                        );
                    if ROW_GROUP == 2 && row + 1 < rows {
                        sum3_b += xs
                            * int_dot16::<KIND>(
                                v0, v1, v2, v3, q0, q1, q2, q3, scale_b, offset0_b, offset1_b,
                            );
                    }
                }
                if MAX > 4 && batch > 4 {
                    let base = (4 * cols + col) / 4;
                    let q0 = unsafe { *x.get_unchecked(base as usize) };
                    let q1 = unsafe { *x.get_unchecked((base + 1) as usize) };
                    let q2 = unsafe { *x.get_unchecked((base + 2) as usize) };
                    let q3 = unsafe { *x.get_unchecked((base + 3) as usize) };
                    let xs = unsafe { *scales.get_unchecked(((4 * cols + col) / 32) as usize) };
                    sum4 += xs
                        * int_dot16::<KIND>(
                            w0, w1, w2, w3, q0, q1, q2, q3, scale, offset0, offset1,
                        );
                    if ROW_GROUP == 2 && row + 1 < rows {
                        sum4_b += xs
                            * int_dot16::<KIND>(
                                v0, v1, v2, v3, q0, q1, q2, q3, scale_b, offset0_b, offset1_b,
                            );
                    }
                }
                if MAX > 5 && batch > 5 {
                    let base = (5 * cols + col) / 4;
                    let q0 = unsafe { *x.get_unchecked(base as usize) };
                    let q1 = unsafe { *x.get_unchecked((base + 1) as usize) };
                    let q2 = unsafe { *x.get_unchecked((base + 2) as usize) };
                    let q3 = unsafe { *x.get_unchecked((base + 3) as usize) };
                    let xs = unsafe { *scales.get_unchecked(((5 * cols + col) / 32) as usize) };
                    sum5 += xs
                        * int_dot16::<KIND>(
                            w0, w1, w2, w3, q0, q1, q2, q3, scale, offset0, offset1,
                        );
                    if ROW_GROUP == 2 && row + 1 < rows {
                        sum5_b += xs
                            * int_dot16::<KIND>(
                                v0, v1, v2, v3, q0, q1, q2, q3, scale_b, offset0_b, offset1_b,
                            );
                    }
                }
                if MAX > 6 && batch > 6 {
                    let base = (6 * cols + col) / 4;
                    let q0 = unsafe { *x.get_unchecked(base as usize) };
                    let q1 = unsafe { *x.get_unchecked((base + 1) as usize) };
                    let q2 = unsafe { *x.get_unchecked((base + 2) as usize) };
                    let q3 = unsafe { *x.get_unchecked((base + 3) as usize) };
                    let xs = unsafe { *scales.get_unchecked(((6 * cols + col) / 32) as usize) };
                    sum6 += xs
                        * int_dot16::<KIND>(
                            w0, w1, w2, w3, q0, q1, q2, q3, scale, offset0, offset1,
                        );
                    if ROW_GROUP == 2 && row + 1 < rows {
                        sum6_b += xs
                            * int_dot16::<KIND>(
                                v0, v1, v2, v3, q0, q1, q2, q3, scale_b, offset0_b, offset1_b,
                            );
                    }
                }
                if MAX > 7 && batch > 7 {
                    let base = (7 * cols + col) / 4;
                    let q0 = unsafe { *x.get_unchecked(base as usize) };
                    let q1 = unsafe { *x.get_unchecked((base + 1) as usize) };
                    let q2 = unsafe { *x.get_unchecked((base + 2) as usize) };
                    let q3 = unsafe { *x.get_unchecked((base + 3) as usize) };
                    let xs = unsafe { *scales.get_unchecked(((7 * cols + col) / 32) as usize) };
                    sum7 += xs
                        * int_dot16::<KIND>(
                            w0, w1, w2, w3, q0, q1, q2, q3, scale, offset0, offset1,
                        );
                    if ROW_GROUP == 2 && row + 1 < rows {
                        sum7_b += xs
                            * int_dot16::<KIND>(
                                v0, v1, v2, v3, q0, q1, q2, q3, scale_b, offset0_b, offset1_b,
                            );
                    }
                }
                col += 512 * SPLIT as u32;
            }
            let mut delta = 16;
            while delta > 0 {
                if MAX > 0 {
                    sum0 += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum0, delta);
                    sum0_b += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum0_b, delta);
                }
                if MAX > 1 {
                    sum1 += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum1, delta);
                    sum1_b += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum1_b, delta);
                }
                if MAX > 2 {
                    sum2 += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum2, delta);
                    sum2_b += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum2_b, delta);
                }
                if MAX > 3 {
                    sum3 += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum3, delta);
                    sum3_b += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum3_b, delta);
                }
                if MAX > 4 {
                    sum4 += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum4, delta);
                    sum4_b += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum4_b, delta);
                }
                if MAX > 5 {
                    sum5 += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum5, delta);
                    sum5_b += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum5_b, delta);
                }
                if MAX > 6 {
                    sum6 += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum6, delta);
                    sum6_b += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum6_b, delta);
                }
                if MAX > 7 {
                    sum7 += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum7, delta);
                    sum7_b += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum7_b, delta);
                }
                delta /= 2;
            }
            if SPLIT > 1 {
                // SAFETY: one complete CTA owns this row, with one writer per
                // warp and token. All threads participate in the barrier.
                let p = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut PARTIAL) };
                if lane == 0 {
                    unsafe {
                        if MAX > 0 && batch > 0 {
                            *p.add(split as usize * 16) = sum0;
                        }
                        if MAX > 1 && batch > 1 {
                            *p.add(split as usize * 16 + 1) = sum1;
                        }
                        if MAX > 2 && batch > 2 {
                            *p.add(split as usize * 16 + 2) = sum2;
                        }
                        if MAX > 3 && batch > 3 {
                            *p.add(split as usize * 16 + 3) = sum3;
                        }
                        if MAX > 4 && batch > 4 {
                            *p.add(split as usize * 16 + 4) = sum4;
                        }
                        if MAX > 5 && batch > 5 {
                            *p.add(split as usize * 16 + 5) = sum5;
                        }
                        if MAX > 6 && batch > 6 {
                            *p.add(split as usize * 16 + 6) = sum6;
                        }
                        if MAX > 7 && batch > 7 {
                            *p.add(split as usize * 16 + 7) = sum7;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 0 && batch > 0 {
                            *p.add(split as usize * 16 + 8) = sum0_b;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 1 && batch > 1 {
                            *p.add(split as usize * 16 + 9) = sum1_b;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 2 && batch > 2 {
                            *p.add(split as usize * 16 + 10) = sum2_b;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 3 && batch > 3 {
                            *p.add(split as usize * 16 + 11) = sum3_b;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 4 && batch > 4 {
                            *p.add(split as usize * 16 + 12) = sum4_b;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 5 && batch > 5 {
                            *p.add(split as usize * 16 + 13) = sum5_b;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 6 && batch > 6 {
                            *p.add(split as usize * 16 + 14) = sum6_b;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 7 && batch > 7 {
                            *p.add(split as usize * 16 + 15) = sum7_b;
                        }
                    }
                }
                thread::sync_threads();
                if split == 0 && lane == 0 {
                    unsafe {
                        if MAX > 0 && batch > 0 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16);
                                j += 1;
                            }
                            *out.add(row as usize) = total;
                        }
                        if MAX > 1 && batch > 1 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 1);
                                j += 1;
                            }
                            *out.add((rows + row) as usize) = total;
                        }
                        if MAX > 2 && batch > 2 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 2);
                                j += 1;
                            }
                            *out.add((2 * rows + row) as usize) = total;
                        }
                        if MAX > 3 && batch > 3 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 3);
                                j += 1;
                            }
                            *out.add((3 * rows + row) as usize) = total;
                        }
                        if MAX > 4 && batch > 4 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 4);
                                j += 1;
                            }
                            *out.add((4 * rows + row) as usize) = total;
                        }
                        if MAX > 5 && batch > 5 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 5);
                                j += 1;
                            }
                            *out.add((5 * rows + row) as usize) = total;
                        }
                        if MAX > 6 && batch > 6 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 6);
                                j += 1;
                            }
                            *out.add((6 * rows + row) as usize) = total;
                        }
                        if MAX > 7 && batch > 7 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 7);
                                j += 1;
                            }
                            *out.add((7 * rows + row) as usize) = total;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 0 && batch > 0 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 8);
                                j += 1;
                            }
                            *out.add((row + 1) as usize) = total;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 1 && batch > 1 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 9);
                                j += 1;
                            }
                            *out.add((rows + row + 1) as usize) = total;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 2 && batch > 2 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 10);
                                j += 1;
                            }
                            *out.add((2 * rows + row + 1) as usize) = total;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 3 && batch > 3 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 11);
                                j += 1;
                            }
                            *out.add((3 * rows + row + 1) as usize) = total;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 4 && batch > 4 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 12);
                                j += 1;
                            }
                            *out.add((4 * rows + row + 1) as usize) = total;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 5 && batch > 5 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 13);
                                j += 1;
                            }
                            *out.add((5 * rows + row + 1) as usize) = total;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 6 && batch > 6 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 14);
                                j += 1;
                            }
                            *out.add((6 * rows + row + 1) as usize) = total;
                        }
                        if ROW_GROUP == 2 && row + 1 < rows && MAX > 7 && batch > 7 {
                            let mut total = 0.0f32;
                            let mut j = 0;
                            while j < SPLIT {
                                total += *p.add(j * 16 + 15);
                                j += 1;
                            }
                            *out.add((7 * rows + row + 1) as usize) = total;
                        }
                    }
                }
                return;
            }
            if lane == 0 {
                unsafe {
                    if MAX > 0 && batch > 0 {
                        *out.add((row) as usize) = sum0;
                    }
                    if MAX > 1 && batch > 1 {
                        *out.add((rows + row) as usize) = sum1;
                    }
                    if MAX > 2 && batch > 2 {
                        *out.add((2 * rows + row) as usize) = sum2;
                    }
                    if MAX > 3 && batch > 3 {
                        *out.add((3 * rows + row) as usize) = sum3;
                    }
                    if MAX > 4 && batch > 4 {
                        *out.add((4 * rows + row) as usize) = sum4;
                    }
                    if MAX > 5 && batch > 5 {
                        *out.add((5 * rows + row) as usize) = sum5;
                    }
                    if MAX > 6 && batch > 6 {
                        *out.add((6 * rows + row) as usize) = sum6;
                    }
                    if MAX > 7 && batch > 7 {
                        *out.add((7 * rows + row) as usize) = sum7;
                    }
                }
                // A paired row reuses the same activation vectors and scales.
                unsafe {
                    if ROW_GROUP == 2 && row + 1 < rows && MAX > 0 && batch > 0 {
                        *out.add((row + 1) as usize) = sum0_b;
                    }
                    if ROW_GROUP == 2 && row + 1 < rows && MAX > 1 && batch > 1 {
                        *out.add((rows + row + 1) as usize) = sum1_b;
                    }
                    if ROW_GROUP == 2 && row + 1 < rows && MAX > 2 && batch > 2 {
                        *out.add((2 * rows + row + 1) as usize) = sum2_b;
                    }
                    if ROW_GROUP == 2 && row + 1 < rows && MAX > 3 && batch > 3 {
                        *out.add((3 * rows + row + 1) as usize) = sum3_b;
                    }
                    if ROW_GROUP == 2 && row + 1 < rows && MAX > 4 && batch > 4 {
                        *out.add((4 * rows + row + 1) as usize) = sum4_b;
                    }
                    if ROW_GROUP == 2 && row + 1 < rows && MAX > 5 && batch > 5 {
                        *out.add((5 * rows + row + 1) as usize) = sum5_b;
                    }
                    if ROW_GROUP == 2 && row + 1 < rows && MAX > 6 && batch > 6 {
                        *out.add((6 * rows + row + 1) as usize) = sum6_b;
                    }
                    if ROW_GROUP == 2 && row + 1 < rows && MAX > 7 && batch > 7 {
                        *out.add((7 * rows + row + 1) as usize) = sum7_b;
                    }
                }
            }
        }
    }

    /// Split the reduction dimension for narrow BF16 gate matrices so a
    /// handful of output rows can occupy the whole GPU.
    #[kernel]
    pub unsafe fn bf16_partial(
        cols: usize,
        rows: usize,
        batch: usize,
        parts: usize,
        weights: &[u8],
        x: &[f32],
        out: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        if group < rows * batch * parts {
            let part = group % parts;
            let row = (group / parts) % rows;
            let token = group / (parts * rows);
            let mut col = part * 256 + lane;
            let mut sum = 0.0f32;
            while col < (part + 1) * 256 && col < cols {
                let off = 2 * (row * cols + col);
                let bits = weights[off] as u32 | ((weights[off + 1] as u32) << 8);
                let w = f32::from_bits(bits << 16);
                sum = w.mul_add(x[token * cols + col], sum);
                col += 32;
            }
            let mut delta = 16;
            while delta > 0 {
                sum += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum, delta);
                delta /= 2;
            }
            if lane == 0 {
                unsafe {
                    *out.add(group) = sum;
                }
            }
        }
    }
    #[kernel]
    pub unsafe fn bf16_merge(parts: usize, input: &[f32], out: *mut f32) {
        let tid = thread::index_1d().get();
        let row = tid / 32;
        let lane = tid % 32;
        if row < input.len() / parts {
            let mut sum = 0.0f32;
            let mut i = lane;
            while i < parts {
                sum += input[row * parts + i];
                i += 32;
            }
            let mut delta = 16;
            while delta > 0 {
                sum += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum, delta);
                delta /= 2;
            }
            if lane == 0 {
                unsafe {
                    *out.add(row) = sum;
                }
            }
        }
    }

    #[kernel]
    pub unsafe fn matmul_eight<const KIND: u32, const BYTES: usize, const ELEMS: usize>(
        cols: usize,
        rows: usize,
        batch: usize,
        weights: &[u8],
        x: &[f32],
        out: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let row = tid / 32;
        let lane = tid % 32;
        if row < rows {
            let mut sum0 = 0.0f32;
            let mut sum1 = 0.0f32;
            let mut sum2 = 0.0f32;
            let mut sum3 = 0.0f32;
            let mut sum4 = 0.0f32;
            let mut sum5 = 0.0f32;
            let mut sum6 = 0.0f32;
            let mut sum7 = 0.0f32;
            let mut col = lane;
            while col < cols {
                let block = (row * cols + col) / ELEMS;
                let w = crate::quant::decode_value(
                    KIND,
                    &weights[block * BYTES..(block + 1) * BYTES],
                    col % ELEMS,
                );
                if batch > 0 {
                    sum0 += w * x[col];
                }
                if batch > 1 {
                    sum1 += w * x[cols + col];
                }
                if batch > 2 {
                    sum2 += w * x[2 * cols + col];
                }
                if batch > 3 {
                    sum3 += w * x[3 * cols + col];
                }
                if batch > 4 {
                    sum4 += w * x[4 * cols + col];
                }
                if batch > 5 {
                    sum5 += w * x[5 * cols + col];
                }
                if batch > 6 {
                    sum6 += w * x[6 * cols + col];
                }
                if batch > 7 {
                    sum7 += w * x[7 * cols + col];
                }
                col += 32;
            }
            let mut delta = 16;
            while delta > 0 {
                sum0 += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum0, delta);
                sum1 += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum1, delta);
                sum2 += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum2, delta);
                sum3 += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum3, delta);
                sum4 += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum4, delta);
                sum5 += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum5, delta);
                sum6 += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum6, delta);
                sum7 += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum7, delta);
                delta /= 2;
            }
            if lane == 0 {
                unsafe {
                    if batch > 0 {
                        *out.add(row) = sum0;
                    }
                    if batch > 1 {
                        *out.add(rows + row) = sum1;
                    }
                    if batch > 2 {
                        *out.add(2 * rows + row) = sum2;
                    }
                    if batch > 3 {
                        *out.add(3 * rows + row) = sum3;
                    }
                    if batch > 4 {
                        *out.add(4 * rows + row) = sum4;
                    }
                    if batch > 5 {
                        *out.add(5 * rows + row) = sum5;
                    }
                    if batch > 6 {
                        *out.add(6 * rows + row) = sum6;
                    }
                    if batch > 7 {
                        *out.add(7 * rows + row) = sum7;
                    }
                }
            }
        }
    }

    #[kernel]
    pub fn embedding_many(
        kind: u32,
        bytes: usize,
        elems: usize,
        dim: usize,
        tokens: &[u32],
        weights: &[u8],
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = out.get_mut(idx) {
            let token = tokens[i / dim] as usize;
            let j = i % dim;
            let block = (token * dim + j) / elems;
            *value = crate::quant::decode_value(
                kind,
                &weights[block * bytes..(block + 1) * bytes],
                j % elems,
            );
        }
    }

    #[kernel]
    pub fn shift_hidden(
        dim: usize,
        previous: &[f32],
        current: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = out.get_mut(idx) {
            *value = if i < dim {
                previous[i]
            } else {
                current[i - dim]
            };
        }
    }

    #[kernel]
    pub fn last_hidden(dim: usize, current: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = out.get_mut(idx) {
            *value = current[current.len() - dim + i];
        }
    }

    #[kernel]
    pub fn concat_rows(dim: usize, a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = out.get_mut(idx) {
            let t = i / (2 * dim);
            let j = i % (2 * dim);
            *value = if j < dim {
                a[t * dim + j]
            } else {
                b[t * dim + j - dim]
            };
        }
    }

    #[kernel]
    pub unsafe fn matvec_packed<const KIND: u32, const BYTES: usize, const ELEMS: usize>(
        cols: usize,
        rows: usize,
        weights: &[u8],
        x: &[f32],
        out: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let row = tid / 32;
        let lane = tid % 32;
        if row < rows {
            let mut sum = 0.0f32;
            if ELEMS == 1 {
                let mut col = lane;
                while col < cols {
                    let block = row * cols + col;
                    sum += crate::quant::decode_value(
                        KIND,
                        &weights[block * BYTES..(block + 1) * BYTES],
                        0,
                    ) * x[col];
                    col += 32;
                }
            } else {
                let mut col = lane * 8;
                while col < cols {
                    let block = (row * cols + col) / ELEMS;
                    let data = &weights[block * BYTES..(block + 1) * BYTES];
                    let i = col % ELEMS;
                    sum += crate::quant::decode_value(KIND, data, i) * x[col];
                    sum += crate::quant::decode_value(KIND, data, i + 1) * x[col + 1];
                    sum += crate::quant::decode_value(KIND, data, i + 2) * x[col + 2];
                    sum += crate::quant::decode_value(KIND, data, i + 3) * x[col + 3];
                    sum += crate::quant::decode_value(KIND, data, i + 4) * x[col + 4];
                    sum += crate::quant::decode_value(KIND, data, i + 5) * x[col + 5];
                    sum += crate::quant::decode_value(KIND, data, i + 6) * x[col + 6];
                    sum += crate::quant::decode_value(KIND, data, i + 7) * x[col + 7];
                    col += 256;
                }
            }
            let mut delta = 16;
            while delta > 0 {
                sum += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum, delta);
                delta /= 2;
            }
            if lane == 0 {
                unsafe {
                    *out.add(row) = sum;
                }
            }
        }
    }

    #[kernel]
    pub unsafe fn matvec(
        kind: u32,
        block_bytes: usize,
        block_elems: usize,
        cols: usize,
        rows: usize,
        weights: &[u8],
        x: &[f32],
        out: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let row = tid / 32;
        let lane = tid % 32;
        if row < rows {
            let mut sum = 0.0f32;
            let mut col = lane;
            while col < cols {
                let block = (row * cols + col) / block_elems;
                let w = crate::quant::decode_value(
                    kind,
                    &weights[block * block_bytes..(block + 1) * block_bytes],
                    col % block_elems,
                );
                sum += w * x[col];
                col += 32;
            }
            let mut delta = 16;
            while delta > 0 {
                sum += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum, delta);
                delta /= 2;
            }
            if lane == 0 {
                unsafe {
                    *out.add(row) = sum;
                }
            }
        }
    }

    #[kernel]
    pub unsafe fn rms_norm(
        dim: usize,
        groups: usize,
        eps: f32,
        x: &[f32],
        weight: &[f32],
        out: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        if group < groups {
            let mut sum = 0.0f32;
            let mut j = lane;
            while j < dim {
                let v = x[group * dim + j];
                sum += v * v;
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
                unsafe {
                    *out.add(group * dim + j) = x[group * dim + j] * scale * weight[j];
                }
                j += 32;
            }
        }
    }

    #[kernel]
    pub fn silu_mul(x: &[f32], gate: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = out.get_mut(idx) {
            *value = x[i] * gate[i] / (1.0 + (-gate[i]).exp());
        }
    }

    #[kernel]
    pub fn decode(kind: u32, block_bytes: usize, input: &[u8], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = out.get_mut(idx) {
            let block = i / 256;
            *value = crate::quant::decode_value(
                kind,
                &input[block * block_bytes..(block + 1) * block_bytes],
                i % 256,
            );
        }
    }

    #[kernel]
    pub fn residual(x: &[f32], y: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = out.get_mut(idx) {
            *value = x[i] + y[i];
        }
    }
}

pub fn check() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let n = 1003;
    let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.125 - 63.0).collect();
    let y: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * -0.25).collect();
    let dx = DeviceBuffer::from_host(&stream, &x)?;
    let dy = DeviceBuffer::from_host(&stream, &y)?;
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, n)?;
    let module = kernels::load(&context)?;
    // SAFETY: the one-dimensional grid covers the output; the kernel guards its
    // tail and every input has exactly the output length.
    unsafe {
        module.residual(
            &stream,
            LaunchConfig::for_num_elems(n as u32),
            &dx,
            &dy,
            &mut out,
        )?;
    }
    let actual = out.to_host_vec(&stream)?;
    for (i, actual) in actual.iter().enumerate() {
        if *actual != x[i] + y[i] {
            return Err(format!("GPU residual mismatch at {i}").into());
        }
    }
    println!("GPU residual: {n} elements match the Rust CPU reference exactly");
    let formats = [
        (10, 84),
        (12, 144),
        (14, 210),
        (16, 66),
        (17, 74),
        (18, 98),
        (19, 50),
        (21, 110),
        (22, 82),
        (23, 136),
        (29, 56),
    ];
    for (kind, block_bytes) in formats {
        let mut input: Vec<u8> = (0..block_bytes * 3)
            .map(|i| ((i * 73 + 19) % 256) as u8)
            .collect();
        for block in input.chunks_exact_mut(block_bytes) {
            let offset = if kind == 10 {
                80
            } else if kind == 14 {
                208
            } else {
                0
            };
            if kind != 29 {
                block[offset..offset + 2].copy_from_slice(&0x3800u16.to_le_bytes());
                if kind == 10 || kind == 12 {
                    block[offset + 2..offset + 4].copy_from_slice(&0x3400u16.to_le_bytes());
                }
            } else {
                for (j, nibble) in [0u16, 0, 8, 3].iter().enumerate() {
                    let pos = 48 + j * 2;
                    let bits = u16::from_le_bytes([block[pos], block[pos + 1]]);
                    block[pos..pos + 2]
                        .copy_from_slice(&((bits & 4095) | (nibble << 12)).to_le_bytes());
                }
            }
        }
        let mut expected = Vec::new();
        for block in input.chunks_exact(block_bytes) {
            expected.extend(crate::quant::decode_block(kind, block)?);
        }
        let din = DeviceBuffer::from_host(&stream, &input)?;
        let mut dout = DeviceBuffer::<f32>::zeroed(&stream, expected.len())?;
        // SAFETY: input has three complete blocks, output has three groups of 256;
        // the one-dimensional grid matches the guarded output index.
        unsafe {
            module.decode(
                &stream,
                LaunchConfig::for_num_elems(expected.len() as u32),
                kind,
                block_bytes,
                &din,
                &mut dout,
            )?;
        }
        let actual = dout.to_host_vec(&stream)?;
        let mut max_error = 0.0f32;
        for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
            let error = (a - e).abs();
            if !a.is_finite() || !e.is_finite() || error > 1e-5 * e.abs().max(1.0) {
                return Err(format!("GPU storage type {kind} mismatch at {i}: {a} vs {e}").into());
            }
            max_error = max_error.max(error);
        }
        println!(
            "GPU storage type {kind}: {} values, maximum error {max_error}",
            expected.len()
        );
        let x: Vec<f32> = (0..256).map(|j| (j as f32 - 117.0) * 0.001).collect();
        let dx = DeviceBuffer::from_host(&stream, &x)?;
        let dout = DeviceBuffer::<f32>::zeroed(&stream, 3)?;
        // SAFETY: one complete warp per row, all inputs cover three 256-element
        // rows, and each row is written once by its warp's lane zero.
        unsafe {
            module.matvec(
                &stream,
                LaunchConfig::for_num_elems(96),
                kind,
                block_bytes,
                256,
                256,
                3,
                &din,
                &dx,
                dout.cu_deviceptr() as *mut f32,
            )?;
        }
        let actual = dout.to_host_vec(&stream)?;
        for (row, a) in actual.iter().enumerate() {
            let expected: f64 = expected[row * 256..(row + 1) * 256]
                .iter()
                .zip(&x)
                .map(|(w, x)| *w as f64 * *x as f64)
                .sum();
            if (*a as f64 - expected).abs() > 2e-4 * expected.abs().max(1.0) {
                return Err(format!("GPU matvec type {kind}, row {row}: {a} vs {expected}").into());
            }
        }
        println!("GPU matvec type {kind}: three rows match f64 reference");
    }
    let dim = 65;
    let groups = 3;
    let input: Vec<f32> = (0..dim * groups)
        .map(|i| (i as f32 - 101.0) * 0.003)
        .collect();
    let weight: Vec<f32> = (0..dim).map(|i| 0.75 + i as f32 * 0.002).collect();
    let dx = DeviceBuffer::from_host(&stream, &input)?;
    let dw = DeviceBuffer::from_host(&stream, &weight)?;
    let output = DeviceBuffer::<f32>::zeroed(&stream, input.len())?;
    // SAFETY: each full warp owns a group; input and output cover three groups,
    // and every lane only writes its own strided elements.
    unsafe {
        module.rms_norm(
            &stream,
            LaunchConfig::for_num_elems((groups * 32) as u32),
            dim,
            groups,
            1e-6,
            &dx,
            &dw,
            output.cu_deviceptr() as *mut f32,
        )?;
    }
    let actual = output.to_host_vec(&stream)?;
    for group in 0..groups {
        let sum: f64 = input[group * dim..(group + 1) * dim]
            .iter()
            .map(|x| (*x as f64).powi(2))
            .sum();
        let scale = 1.0 / (sum / dim as f64 + 1e-6).sqrt();
        for j in 0..dim {
            let expected = input[group * dim + j] as f64 * scale * weight[j] as f64;
            if (actual[group * dim + j] as f64 - expected).abs() > 1e-5 {
                return Err("GPU RMS norm mismatch".into());
            }
        }
    }
    println!("GPU RMS norm: three non-aligned groups match f64 reference");
    Ok(())
}

/// Validate real model matrix rows against an independent f64 host dot product.
pub fn check_model(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::{Read, Seek, SeekFrom};
    let model = crate::gguf::Model::open(path)?;
    let mut file = std::fs::File::open(path)?;
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = kernels::load(&context)?;
    let mut seen = std::collections::BTreeSet::new();
    for tensor in &model.tensors {
        if tensor.shape.len() != 2 || !seen.insert(tensor.kind) {
            continue;
        }
        let (block_elems, block_bytes) =
            crate::gguf::layout(tensor.kind).ok_or("unsupported storage layout")?;
        let cols = tensor.shape[0] as usize;
        let rows = (tensor.shape[1] as usize).min(3);
        let row_bytes = cols / block_elems as usize * block_bytes as usize;
        let mut input = vec![0u8; rows * row_bytes];
        file.seek(SeekFrom::Start(model.data_offset + tensor.offset))?;
        file.read_exact(&mut input)?;
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i * 31 % 127) as f32 - 63.0) * 0.001)
            .collect();
        let integer =
            std::env::var_os("LIA_INT8").is_some() && tensor.kind != 0 && tensor.kind != 30;
        let expected_x: Vec<f32> = if integer {
            let mut values = Vec::new();
            for group in x.as_chunks::<32>().0 {
                let max = group.iter().map(|v| v.abs()).fold(0f32, f32::max);
                let scale = if max == 0.0 { 1.0 } else { max / 127.0 };
                values.extend(group.iter().map(|v| (v / scale).round_ties_even() * scale));
            }
            values
        } else {
            x.clone()
        };
        let mut expected = Vec::new();
        for row in input.chunks_exact(row_bytes) {
            let mut sum = 0.0f64;
            let mut j = 0;
            for block in row.chunks_exact(block_bytes as usize) {
                for value in crate::quant::decode_block(tensor.kind, block)? {
                    sum += value as f64 * expected_x[j] as f64;
                    j += 1;
                }
            }
            expected.push(sum);
        }
        let weights = DeviceBuffer::from_host(&stream, &input)?;
        let dx = DeviceBuffer::from_host(&stream, &x)?;
        let out = DeviceBuffer::<f32>::zeroed(&stream, rows)?;
        // SAFETY: dimensions and byte extents came from the validated container;
        // rows have full storage blocks, and each warp writes its own row.
        if integer {
            let packed = DeviceBuffer::<u32>::zeroed(&stream, cols / 4)?;
            let scales = DeviceBuffer::<f32>::zeroed(&stream, cols / 32)?;
            // SAFETY: the quantized matrix has complete activation groups and
            // correctly sized persistent inputs/outputs through this copy.
            unsafe {
                module.quantize_activation(
                    &stream,
                    LaunchConfig::for_num_elems(cols as u32),
                    &dx,
                    packed.cu_deviceptr() as *mut u32,
                    scales.cu_deviceptr() as *mut f32,
                )?;
                launch_int_matmul(
                    &module,
                    &stream,
                    tensor.kind,
                    cols,
                    rows,
                    1,
                    1,
                    1,
                    &weights,
                    &packed,
                    &scales,
                    out.cu_deviceptr() as *mut f32,
                )?;
            }
            stream.synchronize()?;
        } else {
            unsafe {
                launch_matvec(
                    &module,
                    &stream,
                    tensor.kind,
                    cols,
                    rows,
                    &weights,
                    &dx,
                    out.cu_deviceptr() as *mut f32,
                )?;
            }
        }
        let actual = out.to_host_vec(&stream)?;
        let mut max_error = 0.0f64;
        for (a, e) in actual.iter().zip(&expected) {
            let error = (*a as f64 - *e).abs();
            if !a.is_finite() || error > 1e-4 * e.abs().max(1.0) {
                return Err(format!("model matrix {} mismatch: {a} vs {e}", tensor.name).into());
            }
            max_error = max_error.max(error);
        }
        println!(
            "{} type={} shape={:?} checked_rows={rows} integer={integer} max_error={max_error}",
            tensor.name, tensor.kind, tensor.shape
        );
    }
    Ok(())
}

/// Select a compile-time storage specialization after checking host dimensions.
///
/// # Safety
/// Output must cover `rows` floats and remain alive until this stream completes.
/// Input must cover `cols` floats; weights must contain `rows * cols` elements.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch_matvec(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    kind: u32,
    cols: usize,
    rows: usize,
    weights: &DeviceBuffer<u8>,
    x: &DeviceBuffer<f32>,
    out: *mut f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = LaunchConfig::for_num_elems((rows * 32) as u32);
    let (elems, bytes) = crate::gguf::layout(kind).ok_or("invalid matrix storage type")?;
    if x.len() != cols
        || !cols.is_multiple_of(elems as usize)
        || weights.len() != rows * (cols / elems as usize) * bytes as usize
    {
        return Err("matvec launch shape mismatch".into());
    }
    // SAFETY: caller guarantees output lifetime and extent; checked input shapes
    // and a full-warp one-dimensional launch satisfy each specialization.
    unsafe {
        match kind {
            0 => module.matvec_packed::<0, 4, 1>(stream, config, cols, rows, weights, x, out)?,
            30 => module.matvec_packed::<30, 2, 1>(stream, config, cols, rows, weights, x, out)?,
            10 => {
                module.matvec_packed::<10, 84, 256>(stream, config, cols, rows, weights, x, out)?
            }
            12 => {
                module.matvec_packed::<12, 144, 256>(stream, config, cols, rows, weights, x, out)?
            }
            14 => {
                module.matvec_packed::<14, 210, 256>(stream, config, cols, rows, weights, x, out)?
            }
            16 => {
                module.matvec_packed::<16, 66, 256>(stream, config, cols, rows, weights, x, out)?
            }
            17 => {
                module.matvec_packed::<17, 74, 256>(stream, config, cols, rows, weights, x, out)?
            }
            18 => {
                module.matvec_packed::<18, 98, 256>(stream, config, cols, rows, weights, x, out)?
            }
            19 => {
                module.matvec_packed::<19, 50, 256>(stream, config, cols, rows, weights, x, out)?
            }
            21 => {
                module.matvec_packed::<21, 110, 256>(stream, config, cols, rows, weights, x, out)?
            }
            22 => {
                module.matvec_packed::<22, 82, 256>(stream, config, cols, rows, weights, x, out)?
            }
            23 => {
                module.matvec_packed::<23, 136, 256>(stream, config, cols, rows, weights, x, out)?
            }
            29 => {
                module.matvec_packed::<29, 56, 256>(stream, config, cols, rows, weights, x, out)?
            }
            _ => return Err("unsupported matvec storage type".into()),
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch_matmul(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    kind: u32,
    cols: usize,
    rows: usize,
    batch: usize,
    weights: &DeviceBuffer<u8>,
    x: &DeviceBuffer<f32>,
    out: *mut f32,
) -> Result<(), Box<dyn std::error::Error>> {
    if batch == 1 {
        return unsafe { launch_matvec(module, stream, kind, cols, rows, weights, x, out) };
    }
    let (elems, bytes) = crate::gguf::layout(kind).ok_or("invalid matrix format")?;
    if batch == 0
        || batch > 8
        || x.len() != batch * cols
        || !cols.is_multiple_of(elems as usize)
        || weights.len() != rows * (cols / elems as usize) * bytes as usize
    {
        return Err("batched matrix launch shape mismatch".into());
    }
    let config = LaunchConfig::for_num_elems((rows * 32) as u32);
    // SAFETY: caller owns the output extent and lifetime. Input shapes are
    // checked above; each warp owns one row in every token's output matrix.
    unsafe {
        match kind {
            0 => module
                .matmul_eight::<0, 4, 1>(stream, config, cols, rows, batch, weights, x, out)?,
            30 => module
                .matmul_eight::<30, 2, 1>(stream, config, cols, rows, batch, weights, x, out)?,
            10 => module
                .matmul_eight::<10, 84, 256>(stream, config, cols, rows, batch, weights, x, out)?,
            12 => module
                .matmul_eight::<12, 144, 256>(stream, config, cols, rows, batch, weights, x, out)?,
            14 => module
                .matmul_eight::<14, 210, 256>(stream, config, cols, rows, batch, weights, x, out)?,
            16 => module
                .matmul_eight::<16, 66, 256>(stream, config, cols, rows, batch, weights, x, out)?,
            17 => module
                .matmul_eight::<17, 74, 256>(stream, config, cols, rows, batch, weights, x, out)?,
            18 => module
                .matmul_eight::<18, 98, 256>(stream, config, cols, rows, batch, weights, x, out)?,
            19 => module
                .matmul_eight::<19, 50, 256>(stream, config, cols, rows, batch, weights, x, out)?,
            21 => module
                .matmul_eight::<21, 110, 256>(stream, config, cols, rows, batch, weights, x, out)?,
            22 => module
                .matmul_eight::<22, 82, 256>(stream, config, cols, rows, batch, weights, x, out)?,
            23 => module
                .matmul_eight::<23, 136, 256>(stream, config, cols, rows, batch, weights, x, out)?,
            29 => module
                .matmul_eight::<29, 56, 256>(stream, config, cols, rows, batch, weights, x, out)?,
            _ => return Err("unsupported batched matrix type".into()),
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch_int_matmul(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    kind: u32,
    cols: usize,
    rows: usize,
    batch: usize,
    split: usize,
    row_group: usize,
    weights: &DeviceBuffer<u8>,
    x: &DeviceBuffer<u32>,
    scales: &DeviceBuffer<f32>,
    out: *mut f32,
) -> Result<(), Box<dyn std::error::Error>> {
    if rows.checked_mul(cols).is_none_or(|n| n > u32::MAX as usize)
        || weights.len() > u32::MAX as usize
    {
        return Err("integer kernel index range exceeded".into());
    }
    let (elems, bytes) = crate::gguf::layout(kind).ok_or("invalid integer matrix format")?;
    if batch == 0
        || batch > 8
        || elems != 256
        || !cols.is_multiple_of(256)
        || x.len() != batch * cols / 4
        || (row_group != 1 && row_group != 2)
        || (split == 4 && row_group != 2)
        || (split != 1 && split != 4)
        || scales.len() != batch * cols / 32
        || !(weights.cu_deviceptr() as usize).is_multiple_of(256)
        || weights.len() != rows * (cols / 256) * bytes as usize
    {
        return Err("integer matrix shape mismatch".into());
    }
    let config = if split == 4 {
        LaunchConfig {
            grid_dim: (rows.div_ceil(row_group) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        }
    } else {
        LaunchConfig::for_num_elems((rows.div_ceil(row_group) * 32) as u32)
    };
    // SAFETY: input shapes and full-warp layout were checked above; the caller
    // guarantees the output extent and stream lifetime of all buffers.
    macro_rules! capped {
        ($kind:literal, $bytes:literal, $cap:literal) => {{
            if row_group == 1 {
                module.matmul_int::<$kind, $bytes, $cap, 1, 1>(
                    stream, config, cols, rows, batch, weights, x, scales, out,
                )?;
            } else if split == 4 {
                module.matmul_int::<$kind, $bytes, $cap, 4, 2>(
                    stream, config, cols, rows, batch, weights, x, scales, out,
                )?;
            } else {
                module.matmul_int::<$kind, $bytes, $cap, 1, 2>(
                    stream, config, cols, rows, batch, weights, x, scales, out,
                )?;
            }
        }};
    }
    macro_rules! launch {
        ($kind:literal,$bytes:literal) => {{
            if batch == 1 {
                capped!($kind, $bytes, 1);
            } else if batch <= 4 {
                capped!($kind, $bytes, 4);
            } else {
                capped!($kind, $bytes, 8);
            }
        }};
    }
    // SAFETY: validated matrix extents and one disjoint output row per warp;
    // MAX is a compile-time capacity covering the actual token batch.
    unsafe {
        match kind {
            10 => launch!(10, 84),
            12 => launch!(12, 144),
            14 => launch!(14, 210),
            16 => launch!(16, 66),
            17 => launch!(17, 74),
            18 => launch!(18, 98),
            19 => launch!(19, 50),
            21 => launch!(21, 110),
            22 => launch!(22, 82),
            23 => launch!(23, 136),
            29 => launch!(29, 56),
            _ => return Err("unsupported integer matrix type".into()),
        }
    }
    Ok(())
}

pub fn kernel_info() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let module = kernels::load(&ctx)?;
    let names = [
        (
            "IQ2_XXS/1",
            kernels::matmul_int_ptx_name::<16, 66, 1, 1, 1>(),
        ),
        (
            "IQ2_XXS/4",
            kernels::matmul_int_ptx_name::<16, 66, 4, 1, 1>(),
        ),
        (
            "IQ2_XXS/8",
            kernels::matmul_int_ptx_name::<16, 66, 8, 1, 1>(),
        ),
        ("IQ1_M/1", kernels::matmul_int_ptx_name::<29, 56, 1, 1, 1>()),
        ("IQ1_M/4", kernels::matmul_int_ptx_name::<29, 56, 4, 1, 1>()),
        ("IQ1_M/8", kernels::matmul_int_ptx_name::<29, 56, 8, 1, 1>()),
    ];
    for (label, name) in names {
        let f = module.as_cuda_module().load_function(name)?;
        println!(
            "{label} registers={} local={} blocks_per_sm_256={} blocks_per_sm_128={} blocks_per_sm_64={}",
            f.num_registers()?,
            f.local_size_bytes()?,
            f.max_active_blocks_per_multiprocessor(256, 0)?,
            f.max_active_blocks_per_multiprocessor(128, 0)?,
            f.max_active_blocks_per_multiprocessor(64, 0)?
        );
    }
    Ok(())
}
