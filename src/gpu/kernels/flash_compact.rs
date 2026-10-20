//! Test-only validation of the exact 99-KiB producer/consumer storage layout.
use cuda_device::{kernel, launch_bounds, launch_contract, thread, wmma};
use cuda_host::cuda_module;

pub(crate) const SHARED_BYTES: u32 = 101376;
#[inline(always)]
fn swizzle(row: usize, word: usize) -> usize {
    row * 128 + (word ^ ((row & 7) * 4))
}
#[inline(always)]
fn metadata(index: usize) -> usize {
    (index / 4) * 124 + 120 + index % 4
}

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::super::flash_mma_math::{Float2, Words4, multiply, split};
    use super::*;

    #[inline(always)]
    fn query<const GROUPED: bool>(
        row: usize,
        col: usize,
        qh: usize,
        kvh: usize,
        head: usize,
        start: usize,
        count: usize,
        q: &[f32],
    ) -> [f32; 2] {
        let groups = if GROUPED { qh / kvh } else { 1 };
        let query = start * groups + row;
        if query >= (start + count) * groups {
            return [0.0; 2];
        }
        let h = if GROUPED {
            head * groups + query % groups
        } else {
            head
        };
        unsafe {
            q.as_ptr()
                .add(((query / groups) * qh + h) * 256 + col)
                .cast::<Float2>()
                .read()
                .values
        }
    }
    #[inline(always)]
    fn low(a: f32, b: f32) -> u32 {
        let high = cuda_device::convert::cvt_f16x2_f32(a, b);
        let (x, y) = cuda_device::convert::cvt_f32x2_f16x2(high);
        cuda_device::convert::cvt_f16x2_f32(a - x, b - y)
    }
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(domain=1,dynamic_shared_range=(0,101376),dynamic_shared_alignment=16)]
    pub unsafe fn qk<const COMPACT: bool, const GROUPED: bool>(
        qh: usize,
        kvh: usize,
        head: usize,
        start: usize,
        count: usize,
        key_start: usize,
        tokens: usize,
        q: &[f32],
        k: &[u16],
        output: *mut f32,
    ) {
        let local = thread::index_1d().get();
        let lane = local % 32;
        let w = local / 32;
        let memory = cuda_device::DynamicSharedArray::<u32, 16>::get();
        let hi = memory;
        let lo = unsafe { memory.add(if COMPACT { 8192 } else { 8448 }) };
        let kv = unsafe { memory.add(if COMPACT { 16128 } else { 16896 }) };
        let kh = if GROUPED { head } else { head / (qh / kvh) };
        let mut i = local;
        while i < 8192 {
            let row = i / 128;
            let word = i % 128;
            let [a, b] = query::<GROUPED>(row, word * 2, qh, kvh, head, start, count, q);
            if COMPACT {
                let high = cuda_device::convert::cvt_f16x2_f32(a, b);
                unsafe { *hi.add(swizzle(row, word)) = high };
                if word < 120 {
                    unsafe { *lo.add(row * 124 + word) = low(a, b) };
                }
            } else {
                unsafe { split(hi.add(row * 132 + word), lo.add(row * 132 + word), a, b) };
            }
            i += 512;
        }
        if COMPACT && local < 160 {
            unsafe { *lo.add(metadata(local)) = 0x3f12_0000 + local as u32 };
        }
        i = local * 4;
        while i < 4096 {
            let token = key_start + i / 128;
            let word = i % 128;
            let value = if token < tokens {
                unsafe {
                    k.as_ptr()
                        .cast::<u32>()
                        .add((token * kvh + kh) * 128 + word)
                        .cast::<Words4>()
                        .read()
                }
            } else {
                Words4 { words: [0; 4] }
            };
            let at = if COMPACT {
                swizzle(i / 128, word)
            } else {
                i / 128 * 132 + word
            };
            unsafe { kv.add(at).cast::<Words4>().write(value) };
            i += 2048;
        }
        thread::sync_threads();
        let mut sums = [0.0f32; 4];
        if COMPACT {
            let row = w / 4 * 16 + lane / 4;
            let col = 240 + lane % 4 * 2;
            let a = query::<GROUPED>(row, col, qh, kvh, head, start, count, q);
            let b = query::<GROUPED>(row + 8, col, qh, kvh, head, start, count, q);
            let c = query::<GROUPED>(row, col + 8, qh, kvh, head, start, count, q);
            let d = query::<GROUPED>(row + 8, col + 8, qh, kvh, head, start, count, q);
            let tail = [
                low(a[0], a[1]),
                low(b[0], b[1]),
                low(c[0], c[1]),
                low(d[0], d[1]),
            ];
            let mut column = 0;
            while column < 256 {
                let row = w / 4 * 16 + lane % 16;
                let word = column / 2 + lane / 16 * 4;
                let ah = unsafe { wmma::ldmatrix_x4(hi.add(swizzle(row, word))) };
                let al = if column == 240 {
                    tail
                } else {
                    unsafe { wmma::ldmatrix_x4(lo.add(row * 124 + word)) }
                };
                let bv = unsafe {
                    wmma::ldmatrix_x2(
                        kv.add(swizzle(w % 4 * 8 + lane % 8, column / 2 + lane / 8 % 2 * 4)),
                    )
                };
                sums = unsafe { wmma::mma_m16n8k16_f32_f16(sums, al, bv) };
                sums = unsafe { wmma::mma_m16n8k16_f32_f16(sums, ah, bv) };
                column += 16;
            }
        } else {
            sums = unsafe {
                multiply::<256, 264, false>(
                    sums,
                    hi.add(w / 4 * 16 * 132),
                    lo.add(w / 4 * 16 * 132),
                    kv,
                    (w % 4 * 8) as u32,
                    lane as u32,
                )
            };
        }
        let row = lane / 4 + w / 4 * 16;
        let col = w % 4 * 8 + lane % 4 * 2;
        unsafe {
            *output.add(row * 32 + col) = sums[0] / 16.0;
            *output.add(row * 32 + col + 1) = sums[1] / 16.0;
            *output.add((row + 8) * 32 + col) = sums[2] / 16.0;
            *output.add((row + 8) * 32 + col + 1) = sums[3] / 16.0;
        }
        thread::sync_threads();
        if local < 160 {
            unsafe {
                *output.add(2048 + local) = f32::from_bits(if COMPACT {
                    *lo.add(metadata(local))
                } else {
                    0x3f12_0000 + local as u32
                })
            };
        }
    }
}
