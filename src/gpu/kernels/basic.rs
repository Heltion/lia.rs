use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;

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
