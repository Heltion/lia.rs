//! Shared chunk operands, stable decay products, and triangular coefficients.
use super::delta_mma::{Matrix, put, scalar};
use cuda_device::thread;
pub(super) const LOWER: u32 = 10880;
pub(super) const RATIO: u32 = 11136;
pub(super) const DECAY: u32 = 11392;
pub(super) const BETA: u32 = 11408;
pub(super) const PREFIX: u32 = 11424;

#[inline(always)]
pub(super) unsafe fn matrices(tile: *mut u32) -> [Matrix; 7] {
    unsafe {
        [
            Matrix::at(tile, 0, 1088, 136),
            Matrix::at(tile, 2176, 3264, 136),
            Matrix::at(tile, 4352, 4544, 24),
            Matrix::at(tile, 4736, 4928, 24),
            Matrix::at(tile, 5120, 5440, 40),
            Matrix::at(tile, 5760, 8320, 40),
            Matrix::at(tile, 5760, 6080, 40),
        ]
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn prepare(
    tile: *mut u32,
    q: Matrix,
    k: Matrix,
    head: u32,
    k_heads: u32,
    v_heads: u32,
    start: u32,
    valid: u32,
    local: u32,
    qk: &[f32],
    decay: &[f32],
    beta: &[f32],
) {
    #[repr(C, align(8))]
    #[derive(Clone, Copy)]
    struct Floats2 {
        data: [f32; 2],
    }
    let mut i = local;
    while i < 1024 {
        let t = i / 64;
        let d = (i % 64) * 2;
        let mut qv = [0.0; 2];
        let mut kv = [0.0; 2];
        if t < valid {
            let index = (start + t) * 2 * k_heads * 128 + (head % k_heads) * 128 + d;
            // SAFETY: all columns are even, both complete pairs are in the
            //128-wide allocation, and row starts preserve8-byte alignment.
            unsafe {
                qv = qk
                    .as_ptr()
                    .add(index as usize)
                    .cast::<Floats2>()
                    .read()
                    .data;
                kv = qk
                    .as_ptr()
                    .add((index + k_heads * 128) as usize)
                    .cast::<Floats2>()
                    .read()
                    .data;
            }
        }
        unsafe {
            q.store_pair(t, d, qv[0], qv[1]);
            k.store_pair(t, d, kv[0], kv[1]);
        }
        i += 256;
    }
    if local < 16 {
        let (a, b) = if local < valid {
            let index = ((start + local) * v_heads + head) as usize;
            unsafe { (*decay.get_unchecked(index), *beta.get_unchecked(index)) }
        } else {
            (1.0, 0.0)
        };
        unsafe {
            put(tile, DECAY + local, a);
            put(tile, BETA + local, b);
        }
    }
    thread::sync_threads();
    // Interval products avoid division and remain defined for zero or
    // underflowed decays, including the final partial chunk.
    let r = local / 16;
    let c = local % 16;
    let mut factor = if c <= r { 1.0 } else { 0.0 };
    i = c + 1;
    while i <= r {
        factor *= unsafe { scalar(tile, DECAY + i) };
        i += 1;
    }
    unsafe {
        put(tile, RATIO + r * 16 + c, factor);
    }
    if local < 16 {
        let mut prefix = 1.0;
        i = 0;
        while i <= local {
            prefix *= unsafe { scalar(tile, DECAY + i) };
            i += 1;
        }
        unsafe {
            put(tile, PREFIX + local, prefix);
        }
    }
    thread::sync_threads();
}

#[inline(always)]
pub(super) unsafe fn invert(tile: *mut u32, inverse: Matrix, warp: u32, lane: u32) {
    let mut row = warp;
    while row < 16 {
        let mut x = if lane == row {
            1.0
        } else if lane < row {
            -unsafe { scalar(tile, LOWER + row * 16 + lane) }
        } else {
            0.0
        };
        let mut pivot = row;
        while pivot > 1 {
            pivot -= 1;
            let xp = cuda_device::warp::shuffle_f32_sync(0xffff_ffff, x, pivot);
            if lane < pivot {
                x -= xp * unsafe { scalar(tile, LOWER + pivot * 16 + lane) };
            }
        }
        let next = cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, x, 1);
        if lane < 16 && lane.is_multiple_of(2) {
            unsafe {
                inverse.store_pair(row, lane, x, next);
            }
        }
        row += 8;
    }
}
