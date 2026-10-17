use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;
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
