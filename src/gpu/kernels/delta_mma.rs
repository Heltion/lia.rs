//! Exact shared f32 operands with high/residual TF32 matrix products.
use cuda_device::{convert, wmma};

#[derive(Clone, Copy)]
pub(super) struct Matrix {
    even: *mut u32,
    odd: *mut u32,
    stride: u32,
}

impl Matrix {
    #[inline(always)]
    pub(super) unsafe fn at(tile: *mut u32, even: usize, odd: usize, stride: u32) -> Self {
        Self {
            even: unsafe { tile.add(even) },
            odd: unsafe { tile.add(odd) },
            stride,
        }
    }

    #[inline(always)]
    pub(super) unsafe fn store_pair(self, row: u32, col: u32, a: f32, b: f32) {
        let index = ((row * self.stride + col) / 2) as usize;
        // SAFETY: each producer owns an aligned pair in two disjoint planes.
        unsafe {
            self.even.add(index).write(a.to_bits());
            self.odd.add(index).write(b.to_bits());
        }
    }

    #[inline(always)]
    pub(super) unsafe fn value(self, row: u32, col: u32) -> f32 {
        let index = ((row * self.stride + col) / 2) as usize;
        let plane = if col % 2 == 0 { self.even } else { self.odd };
        unsafe { f32::from_bits(*plane.add(index)) }
    }

    #[inline(always)]
    pub(super) unsafe fn read_pair(self, row: u32, col: u32) -> (f32, f32) {
        unsafe { (self.value(row, col), self.value(row, col + 1)) }
    }

    #[inline(always)]
    pub(super) unsafe fn fragment(self, row: u32, col: u32, lane: u32, x: [f32; 4]) {
        let r = row + lane / 4;
        let c = col + (lane % 4) * 2;
        unsafe {
            self.store_pair(r, c, x[0], x[1]);
            self.store_pair(r + 8, c, x[2], x[3]);
        }
    }
}

#[inline(always)]
fn split(value: f32) -> (u32, u32) {
    let high = convert::cvt_rn_tf32_f32(value);
    let low = convert::cvt_rn_tf32_f32(value - f32::from_bits(high));
    (high, low)
}

/// TF32 fragment coordinates follow the PTX mma.m16n8k8 layout. Keep exact
/// shared operands and all four products of their high/residual components.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn product<const K: u32, const TA: bool, const TB: bool>(
    mut sum: [f32; 4],
    a: Matrix,
    b: Matrix,
    row: u32,
    col: u32,
    lane: u32,
) -> [f32; 4] {
    let mut k = 0;
    while k < K {
        let mut ah = [0u32; 4];
        let mut al = [0u32; 4];
        let mut bh = [0u32; 2];
        let mut bl = [0u32; 2];
        let mut i = 0;
        while i < 4 {
            let r = row + lane / 4 + (i % 2) * 8;
            let c = k + lane % 4 + (i / 2) * 4;
            let value = unsafe { if TA { a.value(c, r) } else { a.value(r, c) } };
            (ah[i as usize], al[i as usize]) = split(value);
            i += 1;
        }
        i = 0;
        while i < 2 {
            let r = k + lane % 4 + i * 4;
            let c = col + lane / 4;
            let value = unsafe { if TB { b.value(r, c) } else { b.value(c, r) } };
            (bh[i as usize], bl[i as usize]) = split(value);
            i += 1;
        }
        unsafe {
            sum = wmma::mma_m16n8k8_f32_tf32(sum, al, bl);
            sum = wmma::mma_m16n8k8_f32_tf32(sum, al, bh);
            sum = wmma::mma_m16n8k8_f32_tf32(sum, ah, bl);
            sum = wmma::mma_m16n8k8_f32_tf32(sum, ah, bh);
        }
        k += 8;
    }
    sum
}

#[inline(always)]
pub(super) unsafe fn scalar(tile: *mut u32, index: u32) -> f32 {
    unsafe { *tile.add(index as usize).cast::<f32>() }
}

#[inline(always)]
pub(super) unsafe fn put(tile: *mut u32, index: u32, value: f32) {
    unsafe {
        tile.add(index as usize).cast::<f32>().write(value);
    }
}

#[inline(always)]
pub(super) unsafe fn load_state(
    state: *const f32,
    head: u32,
    value_base: u32,
    fragment: u32,
    lane: u32,
) -> [f32; 4] {
    let row = (fragment / 4) * 16 + lane / 4;
    let col = value_base + (fragment % 4) * 8 + (lane % 4) * 2;
    unsafe {
        [
            *state.add(((head * 128 + col) * 128 + row) as usize),
            *state.add(((head * 128 + col + 1) * 128 + row) as usize),
            *state.add(((head * 128 + col) * 128 + row + 8) as usize),
            *state.add(((head * 128 + col + 1) * 128 + row + 8) as usize),
        ]
    }
}

#[inline(always)]
pub(super) unsafe fn store_state(
    state: *mut f32,
    head: u32,
    value_base: u32,
    fragment: u32,
    lane: u32,
    x: [f32; 4],
) {
    let row = (fragment / 4) * 16 + lane / 4;
    let col = value_base + (fragment % 4) * 8 + (lane % 4) * 2;
    unsafe {
        *state.add(((head * 128 + col) * 128 + row) as usize) = x[0];
        *state.add(((head * 128 + col + 1) * 128 + row) as usize) = x[1];
        *state.add(((head * 128 + col) * 128 + row + 8) as usize) = x[2];
        *state.add(((head * 128 + col + 1) * 128 + row + 8) as usize) = x[3];
    }
}
