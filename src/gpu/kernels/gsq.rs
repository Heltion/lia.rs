//! Exact-width grouped integer weights with fused scaling and multiplication.
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    struct Float4 {
        values: [f32; 4],
    }

    /// A warp loads 128 consecutive coefficients with one packed load per lane.
    #[kernel]
    pub unsafe fn matvec4<const BITS: u32, const CHANNEL: bool>(
        columns: usize,
        rows: usize,
        batch: usize,
        group_shift: u32,
        packed: &[u8],
        scales: &[f32],
        input: &[f32],
        output: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        if group < batch * rows {
            let row = group % rows;
            let token = group / rows;
            // Channel scales are constant for the complete row, including the
            // large vocabulary projection. Grouped scales are loaded per vector.
            let channel_scale = if CHANNEL {
                unsafe { *scales.get_unchecked(row) }
            } else {
                0.0
            };
            let mut column = lane * 4;
            let mut sum = 0.0f32;
            while column < columns {
                let index = row * columns + column;
                // SAFETY: host requires complete 128-value groups and aligned
                // allocations; row/column offsets preserve word/vector alignment.
                let word = if BITS == 8 {
                    (unsafe { packed.as_ptr().add(index).cast::<u32>().read() }) ^ 0x80808080
                } else {
                    (unsafe { *packed.get_unchecked(index / 4) }) as u32
                };
                let x = unsafe {
                    input
                        .as_ptr()
                        .add(token * columns + column)
                        .cast::<Float4>()
                        .read()
                        .values
                };
                let mut dot = 0.0f32;
                let mut i = 0;
                #[unroll]
                while i < 4 {
                    let code = if BITS == 8 {
                        // Flipping the sign bit converts offset-binary bytes to
                        // signed i8 exactly, avoiding four scalar subtractions.
                        ((word >> (i * 8)) as u8 as i8) as i32
                    } else {
                        ((word >> (i * 2)) & 3) as i32 - 2
                    };
                    dot = (code as f32).mul_add(x[i], dot);
                    i += 1;
                }
                let scale = if CHANNEL {
                    channel_scale
                } else {
                    unsafe {
                        *scales
                            .get_unchecked(row * (columns >> group_shift) + (column >> group_shift))
                    }
                };
                sum = dot.mul_add(scale, sum);
                column += 128;
            }
            let mut delta = 16;
            while delta > 0 {
                sum += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum, delta);
                delta /= 2;
            }
            if lane == 0 {
                unsafe {
                    *output.add(group) = sum;
                }
            }
        }
    }

    #[kernel]
    pub unsafe fn matmul<const BITS: u32>(
        columns: usize,
        rows: usize,
        batch: usize,
        group_shift: u32,
        packed: &[u8],
        scales: &[f32],
        input: &[f32],
        output: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        if group < batch * rows {
            let row = group % rows;
            let token = group / rows;
            let per_byte = 8 / BITS as usize;
            let mut column = lane;
            let mut sum = 0.0f32;
            while column < columns {
                let index = row * columns + column;
                let code = ((packed[index / per_byte] as u32)
                    >> ((index % per_byte) * BITS as usize))
                    & ((1 << BITS) - 1);
                let coefficient = code as i32 - (1 << (BITS - 1));
                let scale = scales[row * (columns >> group_shift) + (column >> group_shift)];
                sum = (coefficient as f32 * scale).mul_add(input[token * columns + column], sum);
                column += 32;
            }
            let mut delta = 16;
            while delta > 0 {
                sum += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum, delta);
                delta /= 2;
            }
            if lane == 0 {
                unsafe {
                    *output.add(group) = sum;
                }
            }
        }
    }
}
