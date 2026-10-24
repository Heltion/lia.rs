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
    pub fn silu_joined(rows: usize, input: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = out.get_mut(idx) {
            let offset = i / rows * (2 * rows) + i % rows;
            let gate = input[offset];
            let up = input[offset + rows];
            *value = up * gate / (1.0 + (-gate).exp());
        }
    }

    #[kernel]
    pub fn silu_interleaved(rows: usize, input: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = out.get_mut(idx) {
            let row = i % rows;
            let offset = i / rows * (rows * 2) + row / 8 * 16 + row % 8;
            let gate = input[offset];
            *value = input[offset + 8] * gate / (1.0 + (-gate).exp());
        }
    }

    #[kernel]
    pub fn silu_mul_in_place(gate: &[f32], mut values: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = values.get_mut(idx) {
            *value = *value * gate[i] / (1.0 + (-gate[i]).exp());
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

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};

    #[test]
    fn owned_activation_matches_separate_output() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = crate::gpu::kernels::load(&context)?;
        for count in [1, 31, 257, 17408] {
            let input: Vec<f32> = (0..count).map(|i| (i % 113) as f32 * 0.25 - 13.0).collect();
            let gates: Vec<f32> = (0..count).map(|i| (i % 503) as f32 - 251.0).collect();
            let original = DeviceBuffer::from_host(&stream, &input)?;
            let gate = DeviceBuffer::from_host(&stream, &gates)?;
            let mut owned = DeviceBuffer::from_host(&stream, &input)?;
            let mut separate = DeviceBuffer::<f32>::zeroed(&stream, count)?;
            let config = LaunchConfig::for_num_elems(count as u32);
            // SAFETY: equal logical lengths and distinct owned allocations.
            unsafe {
                module
                    .basic
                    .silu_mul(&stream, config, &original, &gate, &mut separate)?;
                module
                    .basic
                    .silu_mul_in_place(&stream, config, &gate, &mut owned)?;
            }
            let actual = owned.to_host_vec(&stream)?;
            let expected = separate.to_host_vec(&stream)?;
            assert_eq!(
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(original.to_host_vec(&stream)?, input);
        }
        Ok(())
    }
    #[test]
    fn joined_gateup_activation_matches_independent_inputs() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = crate::gpu::kernels::load(&context)?;
        for (rows, batch) in [(1usize, 1usize), (31, 3), (257, 17), (12288, 34)] {
            let n = rows * batch;
            let up: Vec<f32> = (0..n).map(|i| (i % 113) as f32 * 0.25 - 13.0).collect();
            let gate: Vec<f32> = (0..n).map(|i| (i % 503) as f32 - 251.0).collect();
            let mut joined = Vec::with_capacity(n * 2);
            for token in 0..batch {
                joined.extend_from_slice(&gate[token * rows..(token + 1) * rows]);
                joined.extend_from_slice(&up[token * rows..(token + 1) * rows]);
            }
            let input = DeviceBuffer::from_host(&stream, &joined)?;
            let du = DeviceBuffer::from_host(&stream, &up)?;
            let dg = DeviceBuffer::from_host(&stream, &gate)?;
            let mut actual = DeviceBuffer::zeroed(&stream, n)?;
            let mut expected = DeviceBuffer::zeroed(&stream, n)?;
            let config = LaunchConfig::for_num_elems(n as u32);
            // SAFETY: complete gate/up pairs per token; guarded exact outputs.
            unsafe {
                module
                    .basic
                    .silu_joined(&stream, config, rows, &input, &mut actual)?;
                module
                    .basic
                    .silu_mul(&stream, config, &du, &dg, &mut expected)?;
            }
            assert!(
                actual
                    .to_host_vec(&stream)?
                    .iter()
                    .zip(expected.to_host_vec(&stream)?)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "joined activation rows={rows} batch={batch}"
            );
        }
        Ok(())
    }
}
