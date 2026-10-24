//! Warp-owned score fragments reuse Q loads and keep softmax reductions in registers.
use cuda_device::{convert, wmma};
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub(super) struct Words4 {
    pub words: [u32; 4],
}
#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub(super) struct Float2 {
    pub values: [f32; 2],
}
#[inline(always)]
pub(super) fn exp(x: f32) -> f32 {
    cuda_device::float::ex2_approx_f32(x)
}
#[inline(always)]
pub(super) fn pair(a: f32, b: f32) -> u32 {
    convert::cvt_f16x2_f32(a, b)
}

#[inline(always)]
pub(super) unsafe fn qk<const COLUMNS: usize>(
    a: *const u32,
    b: *const u32,
    lane: u32,
) -> [[f32; 4]; 8] {
    let mut result = [[0_f32; 4]; 8];
    let mut step = 0;
    while step < 8 {
        let k = step * 16;
        let ar =
            unsafe { wmma::ldmatrix_x4(a.add((lane % 16 * 68 + k / 2 + lane / 16 * 4) as usize)) };
        macro_rules! product_pair {
            ($j:literal, $next:literal) => {{
                // Four 8x8 matrices carry two complete N8/K16 operands.
                // Lanes 16..31 supply the second operand's row addresses.
                unsafe {
                    let br = wmma::ldmatrix_x4(b.add(
                        (($j * 8 + lane % 8 + lane / 16 * 8) * 68 + k / 2 + lane / 8 % 2 * 4)
                            as usize,
                    ));
                    result[$j] = wmma::mma_m16n8k16_f32_f16(result[$j], ar, [br[0], br[1]]);
                    result[$next] = wmma::mma_m16n8k16_f32_f16(result[$next], ar, [br[2], br[3]]);
                }
            }};
        }
        product_pair!(0, 1);
        product_pair!(2, 3);
        if COLUMNS == 64 {
            product_pair!(4, 5);
            product_pair!(6, 7);
        }
        step += 1;
    }
    result
}

#[inline(always)]
pub(super) unsafe fn pv<const COLUMNS: usize>(
    mut result: [[f32; 4]; 16],
    a: *const u32,
    b: *const u32,
    lane: u32,
) -> [[f32; 4]; 16] {
    let mut step = 0;
    while step < COLUMNS / 16 {
        let k = step as u32 * 16;
        let ar = unsafe {
            wmma::ldmatrix_x4(a.add(
                (lane % 16 * ((COLUMNS + 8) / 2) as u32 + k as u32 / 2 + lane / 16 * 4) as usize,
            ))
        };
        macro_rules! product_pair {
            ($j:literal, $next:literal) => {{
                unsafe {
                    let br = wmma::ldmatrix_x4_trans(
                        b.add(((k + lane % 16) * 68 + $j * 4 + lane / 16 * 4) as usize),
                    );
                    result[$j] = wmma::mma_m16n8k16_f32_f16(result[$j], ar, [br[0], br[1]]);
                    result[$next] = wmma::mma_m16n8k16_f32_f16(result[$next], ar, [br[2], br[3]]);
                }
            }};
        }
        product_pair!(0, 1);
        product_pair!(2, 3);
        product_pair!(4, 5);
        product_pair!(6, 7);
        product_pair!(8, 9);
        product_pair!(10, 11);
        product_pair!(12, 13);
        product_pair!(14, 15);
        step += 1;
    }
    result
}

#[inline(always)]
pub(super) unsafe fn load_tile<const ASYNC: bool, const THREADS: usize, const KV_HEADS: usize>(
    cache: &[u16],
    destination: *mut u32,
    physical: usize,
    head: usize,
    start: usize,
    end: usize,
    local: usize,
) {
    let mut i = local * 4;
    while i < 32 * 64 {
        let row = i / 64;
        let dest = unsafe { destination.add(row * 68 + i % 64) };
        if start + row < end {
            let src = unsafe {
                cache
                    .as_ptr()
                    .add(((physical + row) * KV_HEADS + head) * 128 + i % 64 * 2)
            };
            unsafe {
                if ASYNC {
                    cuda_device::async_copy::cp_async_ca_16(dest, src.cast());
                } else {
                    dest.cast::<Words4>().write(src.cast::<Words4>().read());
                }
            }
        } else {
            unsafe { dest.cast::<Words4>().write(Words4 { words: [0; 4] }) };
        }
        i += THREADS * 4;
    }
}

#[inline(always)]
pub(super) unsafe fn load_queries<
    const ROWS: usize,
    const THREADS: usize,
    const KV_HEADS: usize,
    const GROUP: usize,
>(
    q: &[f32],
    destination: *mut u32,
    base: usize,
    queries: usize,
    head: usize,
    local: usize,
) {
    let mut i = local;
    while i < ROWS * 64 {
        let row = i / 64;
        let column = i % 64 * 2;
        let at = base + row;
        let packed = if at < queries {
            let qhead = head * GROUP + at % GROUP;
            let offset = (at / GROUP * (KV_HEADS * GROUP) + qhead) * 128 + column;
            let values = unsafe { q.as_ptr().add(offset).cast::<Float2>().read().values };
            pair(values[0], values[1])
        } else {
            0
        };
        unsafe {
            destination.add(row * 68 + column / 2).write(packed);
        }
        i += THREADS;
    }
}

#[inline(always)]
pub(super) unsafe fn pv_f32<const COLUMNS: usize>(
    mut output: [[f32; 4]; 16],
    probability: *const f32,
    values: *const u32,
    warp: usize,
    lane: usize,
) -> [[f32; 4]; 16] {
    let row = warp * 16 + lane / 4;
    let mut key = 0;
    while key < COLUMNS {
        let a = unsafe { *probability.add(row * COLUMNS + key) };
        let b = unsafe { *probability.add((row + 8) * COLUMNS + key) };
        macro_rules! column {
            ($j:literal) => {{
                let packed = unsafe { *values.add(key * 68 + $j * 4 + lane % 4) };
                let (x, y) = convert::cvt_f32x2_f16x2(packed);
                output[$j][0] = a.mul_add(x, output[$j][0]);
                output[$j][1] = a.mul_add(y, output[$j][1]);
                output[$j][2] = b.mul_add(x, output[$j][2]);
                output[$j][3] = b.mul_add(y, output[$j][3]);
            }};
        }
        column!(0);
        column!(1);
        column!(2);
        column!(3);
        column!(4);
        column!(5);
        column!(6);
        column!(7);
        column!(8);
        column!(9);
        column!(10);
        column!(11);
        column!(12);
        column!(13);
        column!(14);
        column!(15);
        key += 1;
    }
    output
}
