use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;
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
}
