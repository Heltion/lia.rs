//! Single-piece BF16 matrix operands; accumulator remains FP32.
use cuda_device::wmma;
#[inline(always)]
pub(super) unsafe fn multiply<const K: u32, const STRIDE: u32, const TRANS: bool>(
    mut sum: [f32; 4],
    a: *const u32,
    b: *const u32,
    col: u32,
    lane: u32,
) -> [f32; 4] {
    let mut k = 0;
    while k < K {
        unsafe {
            let offset = ((lane % 16 * STRIDE + k + lane / 16 * 8) / 2) as usize;
            let ah = wmma::ldmatrix_x4(a.add(offset));
            let bv = if TRANS {
                wmma::ldmatrix_x2_trans(b.add(((k + lane % 16) * 132 + col / 2) as usize))
            } else {
                wmma::ldmatrix_x2(
                    b.add(((col + lane % 8) * 132 + k / 2 + lane / 8 % 2 * 4) as usize),
                )
            };
            sum = wmma::mma_m16n8k16_f32_bf16(sum, ah, bv);
        }
        k += 16;
    }
    sum
}
