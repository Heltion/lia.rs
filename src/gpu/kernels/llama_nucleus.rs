//! Exact local candidates and full-vocabulary FP64 softmax mass.
use cuda_device::{kernel, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[inline(always)]
    fn down(value: f64, offset: u32) -> f64 {
        let bits = value.to_bits();
        let low = warp::shuffle_down_sync(0xffff_ffff, bits as u32, offset);
        let high = warp::shuffle_down_sync(0xffff_ffff, (bits >> 32) as u32, offset);
        f64::from_bits((u64::from(high) << 32) | u64::from(low))
    }

    #[kernel]
    pub unsafe fn llama_nucleus_candidates(
        vocabulary: usize,
        parts: usize,
        slots: usize,
        temperatures: &[f64],
        logits: &[f32],
        candidates: *mut u64,
        normalizers: *mut f64,
    ) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        if group < logits.len() / vocabulary * parts {
            let row = group / parts;
            let part = group % parts;
            let mut values = [f32::NEG_INFINITY; 32];
            let mut maximum = f32::NEG_INFINITY;
            let mut invalid = false;
            let mut j = 0;
            #[unroll]
            while j < 32 {
                let id = part * 1024 + lane + j * 32;
                if id < vocabulary {
                    let value = logits[row * vocabulary + id];
                    values[j] = value;
                    maximum = maximum.max(value);
                    invalid |= value.is_nan() || value == f32::INFINITY;
                }
                j += 1;
            }
            let mut offset = 16;
            while offset > 0 {
                maximum = maximum.max(warp::shuffle_down_f32_sync(0xffff_ffff, maximum, offset));
                offset /= 2;
            }
            maximum = warp::shuffle_f32_sync(0xffff_ffff, maximum, 0);
            let mut sum = if invalid { f64::NAN } else { 0.0 };
            let temperature = temperatures[row];
            if temperature > 0.0 && maximum.is_finite() {
                j = 0;
                while j < 32 {
                    if values[j].is_finite() {
                        sum += ((f64::from(values[j]) - f64::from(maximum)) / temperature).exp();
                    }
                    j += 1;
                }
            }
            offset = 16;
            while offset > 0 {
                sum += down(sum, offset as u32);
                offset /= 2;
            }
            if lane == 0 {
                unsafe {
                    *normalizers.add(group * 2) = f64::from(maximum);
                    *normalizers.add(group * 2 + 1) = sum;
                }
            }
            let mut used = 0_u32;
            let mut rank = 0;
            while rank < slots {
                let mut best = f32::NEG_INFINITY;
                let mut best_key = i32::MIN;
                let mut best_id = u32::MAX;
                j = 0;
                #[unroll]
                while j < 32 {
                    let id = part * 1024 + lane + j * 32;
                    let bits = values[j].to_bits() as i32;
                    let key = bits ^ (((bits >> 31) as u32 >> 1) as i32);
                    if id < vocabulary
                        && used & (1 << j) == 0
                        && (key > best_key || (key == best_key && id as u32 > best_id))
                    {
                        best = values[j];
                        best_key = key;
                        best_id = id as u32;
                    }
                    j += 1;
                }
                offset = 16;
                while offset > 0 {
                    let other = warp::shuffle_down_f32_sync(0xffff_ffff, best, offset as u32);
                    let id = warp::shuffle_down_sync(0xffff_ffff, best_id, offset as u32);
                    let bits = other.to_bits() as i32;
                    let key = bits ^ (((bits >> 31) as u32 >> 1) as i32);
                    if id != u32::MAX && (key > best_key || (key == best_key && id > best_id)) {
                        best = other;
                        best_key = key;
                        best_id = id;
                    }
                    offset /= 2;
                }
                let winner = warp::shuffle_sync(0xffff_ffff, best_id, 0);
                if lane == 0 {
                    unsafe {
                        *candidates.add(group * slots + rank) =
                            (u64::from(winner) << 32) | u64::from(best.to_bits());
                    }
                }
                if winner != u32::MAX && winner % 32 == lane as u32 {
                    used |= 1 << ((winner as usize - part * 1024) / 32);
                }
                rank += 1;
            }
        }
    }
}
