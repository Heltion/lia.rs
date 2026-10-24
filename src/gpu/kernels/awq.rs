//! Fused explicit-zero AWQ multiplication without a dense weight workspace.
use cuda_device::{kernel, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;

    /// Host validates complete warps, group multiples of 32 and output extents.
    #[kernel]
    pub unsafe fn awq_word8(
        columns: usize,
        rows: usize,
        group_size: usize,
        batch: usize,
        parts: usize,
        packed: &[u32],
        zeros: &[u32],
        scales: &[f32],
        input: &[f32],
        output: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let word_row = tid / 32;
        let lane = tid % 32;
        let words_per_token = rows / 8;
        if word_row < batch * parts * words_per_token {
            let token_part = word_row / words_per_token;
            let token = token_part / parts;
            let part = token_part % parts;
            let word_row = word_row % words_per_token;
            let groups = columns / group_size;
            let mut sums = [0.0f32; 8];
            let mut group = part * groups / parts;
            let end = (part + 1) * groups / parts;
            while group < end {
                let zero_word = unsafe { *zeros.get_unchecked(word_row * groups + group) };
                let mut gains = [0.0f32; 8];
                let mut row = 0;
                #[unroll]
                while row < 8 {
                    gains[row] =
                        unsafe { *scales.get_unchecked(group * rows + word_row * 8 + row) };
                    row += 1;
                }
                let mut column = group * group_size + lane;
                while column < (group + 1) * group_size {
                    let weight = unsafe { *packed.get_unchecked(word_row * columns + column) };
                    let x = unsafe { *input.get_unchecked(token * columns + column) };
                    let mut row = 0;
                    #[unroll]
                    while row < 8 {
                        let shift = ((row % 2) * 4 + row / 2) * 4;
                        let code = ((weight >> shift) & 15) as i32;
                        let zero = ((zero_word >> shift) & 15) as i32;
                        // Dequantize before accumulation: moving the group scale
                        // outside the dot product changes Float32 rounding.
                        let value = (code - zero) as f32 * gains[row];
                        sums[row] = value.mul_add(x, sums[row]);
                        row += 1;
                    }
                    column += 32;
                }
                group += 1;
            }
            let mut row = 0;
            #[unroll]
            while row < 8 {
                let mut value = sums[row];
                let mut delta = 16;
                while delta > 0 {
                    value += warp::shuffle_down_f32_sync(0xffff_ffff, value, delta);
                    delta /= 2;
                }
                if lane == 0 {
                    unsafe {
                        *output.add(token_part * rows + word_row * 8 + row) = value;
                    }
                }
                row += 1;
            }
        }
    }

    /// Disjoint output rows combine complete group partitions in a fixed order.
    #[kernel]
    pub unsafe fn awq_merge(
        rows: usize,
        batch: usize,
        parts: usize,
        partial: &[f32],
        output: *mut f32,
    ) {
        let i = thread::index_1d().get();
        if i < batch * rows {
            let mut value = 0.0f32;
            let mut part = 0;
            while part < parts {
                value += partial[((i / rows) * parts + part) * rows + i % rows];
                part += 1;
            }
            unsafe {
                *output.add(i) = value;
            }
        }
    }
}
