//! Shared BF16 high/residual operands for f32 recurrent matrix products.
use cuda_device::{convert, wmma};

#[derive(Clone, Copy)]
pub(super) struct Matrix {
    hi: *mut u32,
    lo: *mut u32,
    stride: u32,
}

impl Matrix {
    #[inline(always)]
    pub(super) unsafe fn at(tile: *mut u32, hi: usize, lo: usize, stride: u32) -> Self {
        Self {
            hi: unsafe { tile.add(hi) },
            lo: unsafe { tile.add(lo) },
            stride,
        }
    }

    #[inline(always)]
    pub(super) unsafe fn store_pair(self, row: u32, col: u32, a: f32, b: f32) {
        let hi = convert::cvt_bf16x2_f32(a, b);
        let (ah, bh) = convert::cvt_f32x2_bf16x2(hi);
        let lo = convert::cvt_bf16x2_f32(a - ah, b - bh);
        let index = ((row * self.stride + col) / 2) as usize;
        // SAFETY: each producer owns a complete aligned pair in both planes.
        unsafe {
            self.hi.add(index).write(hi);
            self.lo.add(index).write(lo);
        }
    }

    #[inline(always)]
    pub(super) unsafe fn read_pair(self, row: u32, col: u32) -> (f32, f32) {
        let index = ((row * self.stride + col) / 2) as usize;
        let (ah, bh) = unsafe { convert::cvt_f32x2_bf16x2(*self.hi.add(index)) };
        let (al, bl) = unsafe { convert::cvt_f32x2_bf16x2(*self.lo.add(index)) };
        (ah + al, bh + bl)
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
unsafe fn load_a<const TRANS: bool>(
    matrix: Matrix,
    low: bool,
    row: u32,
    k: u32,
    lane: u32,
) -> [u32; 4] {
    let plane = if low { matrix.lo } else { matrix.hi };
    let (r, c) = if TRANS {
        (k + lane % 8 + (lane / 16) * 8, row + ((lane / 8) % 2) * 8)
    } else {
        (row + lane % 16, k + (lane / 16) * 8)
    };
    let ptr = unsafe { plane.add(((r * matrix.stride + c) / 2) as usize) };
    unsafe {
        if TRANS {
            wmma::ldmatrix_x4_trans(ptr)
        } else {
            wmma::ldmatrix_x4(ptr)
        }
    }
}

#[inline(always)]
unsafe fn load_b<const TRANS: bool>(
    matrix: Matrix,
    low: bool,
    col: u32,
    k: u32,
    lane: u32,
) -> [u32; 2] {
    let plane = if low { matrix.lo } else { matrix.hi };
    let (r, c) = if TRANS {
        (k + lane % 16, col)
    } else {
        (col + lane % 8, k + ((lane / 8) % 2) * 8)
    };
    let ptr = unsafe { plane.add(((r * matrix.stride + c) / 2) as usize) };
    unsafe {
        if TRANS {
            wmma::ldmatrix_x2_trans(ptr)
        } else {
            wmma::ldmatrix_x2(ptr)
        }
    }
}

/// B is addressed as B-transpose unless TB is true. Keep both first-order
/// residual products, with the persistent accumulator always remaining f32.
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
        unsafe {
            let ah = load_a::<TA>(a, false, row, k, lane);
            let al = load_a::<TA>(a, true, row, k, lane);
            let bh = load_b::<TB>(b, false, col, k, lane);
            let bl = load_b::<TB>(b, true, col, k, lane);
            sum = wmma::mma_m16n8k16_f32_bf16(sum, al, bh);
            sum = wmma::mma_m16n8k16_f32_bf16(sum, ah, bl);
            sum = wmma::mma_m16n8k16_f32_bf16(sum, ah, bh);
        }
        k += 16;
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
