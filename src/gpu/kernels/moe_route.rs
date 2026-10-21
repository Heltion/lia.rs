//! Exact top-eight router selection shared by standalone and fused launches.
macro_rules! select {
    ($values:expr,$lane:expr,$token:expr,$selected:expr,$probabilities:expr) => {{
        let mut values = $values;
        let lane = $lane;
        let token = $token;
        let selected = $selected;
        let probabilities = $probabilities;
        let mut scores = [0.0f32; 8];
        let mut sum = 0.0;
        let mut maximum = 0.0;
        let mut rank = 0;
        let mut j;
        while rank < 8 {
            let mut best = f32::NEG_INFINITY;
            let mut id = u32::MAX;
            j = 0;
            while j < 8 {
                let candidate = (lane + j * 32) as u32;
                if values[j] > best
                    || (best != f32::NEG_INFINITY && values[j] == best && candidate < id)
                {
                    best = values[j];
                    id = candidate;
                }
                j += 1;
            }
            let mut delta = 16;
            while delta > 0 {
                let other = cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, best, delta);
                let other_id = cuda_device::warp::shuffle_down_sync(0xffff_ffff, id, delta);
                if other > best || (other == best && other_id < id) {
                    best = other;
                    id = other_id;
                }
                delta /= 2;
            }
            let winner = cuda_device::warp::shuffle_sync(0xffff_ffff, id, 0);
            if winner < 256 && winner as usize % 32 == lane {
                values[winner as usize / 32] = f32::NEG_INFINITY;
            }
            if lane == 0 {
                if rank == 0 {
                    maximum = best;
                }
                scores[rank] = (best - maximum).exp();
                sum += scores[rank];
                // SAFETY: each full warp owns eight disjoint routing slots.
                unsafe { *selected.add(token * 8 + rank) = winner };
            }
            rank += 1;
        }
        if lane == 0 {
            rank = 0;
            while rank < 8 {
                unsafe { *probabilities.add(token * 8 + rank) = scores[rank] / sum };
                rank += 1;
            }
        }
    }};
}
pub(super) use select;
