//! MMA probability fragments map directly from softmax accumulators to A registers.
use cuda_device::{convert, wmma};

#[inline(always)]
fn fragment<const LOW: bool>(a: f32, b: f32) -> u32 {
    let high = convert::cvt_f16x2_f32(a, b);
    if LOW {
        let (ah, bh) = convert::cvt_f32x2_f16x2(high);
        convert::cvt_f16x2_f32(a - ah, b - bh)
    } else {
        high
    }
}

#[inline(always)]
pub(super) unsafe fn pv_register<const COLUMNS: usize>(
    mut result: [[f32; 4]; 16],
    probability: [[f32; 4]; 8],
    values: *const u32,
    lane: u32,
) -> [[f32; 4]; 16] {
    macro_rules! step {
        ($first:literal, $second:literal, $offset:literal, $low:literal) => {{
            let a0 = fragment::<$low>(probability[$first][0], probability[$first][1]);
            let a1 = fragment::<$low>(probability[$first][2], probability[$first][3]);
            let a2 = fragment::<$low>(probability[$second][0], probability[$second][1]);
            let a3 = fragment::<$low>(probability[$second][2], probability[$second][3]);
            let high = [a0, a1, a2, a3];
            macro_rules! product_pair {
                ($j:literal, $next:literal) => {{
                    unsafe {
                        let br =
                            wmma::ldmatrix_x4_trans(values.add(
                                (($offset + lane % 16) * 68 + $j * 4 + lane / 16 * 4) as usize,
                            ));
                        result[$j] = wmma::mma_m16n8k16_f32_f16(result[$j], high, [br[0], br[1]]);
                        result[$next] =
                            wmma::mma_m16n8k16_f32_f16(result[$next], high, [br[2], br[3]]);
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
        }};
    }
    // Match shared residual-PV exactly: finish every high product before
    // adding any residual products to the existing FP32 accumulator.
    step!(0, 1, 0, false);
    step!(2, 3, 16, false);
    if COLUMNS == 64 {
        step!(4, 5, 32, false);
        step!(6, 7, 48, false);
    }
    step!(0, 1, 0, true);
    step!(2, 3, 16, true);
    if COLUMNS == 64 {
        step!(4, 5, 32, true);
        step!(6, 7, 48, true);
    }
    result
}
