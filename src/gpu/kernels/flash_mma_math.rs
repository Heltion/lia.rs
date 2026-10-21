//! Shared packed loads, query residual splitting and ordered MMA operations.
use cuda_device::{convert, wmma};

#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub(super) struct Words4 {
    pub(super) words: [u32; 4],
}

#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub(super) struct Float2 {
    pub(super) values: [f32; 2],
}

#[inline(always)]
pub(super) fn exp(x: f32) -> f32 {
    cuda_device::float::ex2_approx_f32(x * core::f32::consts::LOG2_E)
}

#[inline(always)]
pub(super) unsafe fn split(dest: *mut u32, low: *mut u32, a: f32, b: f32) {
    let hi = convert::cvt_f16x2_f32(a, b);
    let (ah, bh) = convert::cvt_f32x2_f16x2(hi);
    unsafe {
        dest.write(hi);
        low.write(convert::cvt_f16x2_f32(a - ah, b - bh));
    }
}

#[inline(always)]
pub(super) unsafe fn multiply<const K: u32, const STRIDE: u32, const TRANS: bool>(
    mut sum: [f32; 4],
    a: *const u32,
    low: *const u32,
    b: *const u32,
    col: u32,
    lane: u32,
) -> [f32; 4] {
    let mut k = 0;
    while k < K {
        unsafe {
            let offset = ((lane % 16 * STRIDE + k + lane / 16 * 8) / 2) as usize;
            let ah = wmma::ldmatrix_x4(a.add(offset));
            let al = wmma::ldmatrix_x4(low.add(offset));
            let bv = if TRANS {
                wmma::ldmatrix_x2_trans(b.add(((k + lane % 16) * 132 + col / 2) as usize))
            } else {
                wmma::ldmatrix_x2(
                    b.add(((col + lane % 8) * 132 + k / 2 + lane / 8 % 2 * 4) as usize),
                )
            };
            sum = wmma::mma_m16n8k16_f32_f16(sum, al, bv);
            sum = wmma::mma_m16n8k16_f32_f16(sum, ah, bv);
        }
        k += 16;
    }
    sum
}
