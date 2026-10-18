//! GPU routing and selected 2-bit experts; no host round-trip for expert IDs.
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    struct Float4 {
        values: [f32; 4],
    }

    #[inline(always)]
    fn coefficients<const HOST: bool>(
        packed: &[u8],
        scales: &[u16],
        host: *const u8,
        host_scales: *const u16,
        index: usize,
    ) -> (u8, f32) {
        let (byte, scale) = if !HOST || index / 4 < packed.len() {
            (packed[index / 4], scales[index / 128])
        } else {
            // SAFETY: the mapped immutable suffix covers all nonresident rows.
            unsafe {
                (
                    *host.add(index / 4 - packed.len()),
                    *host_scales.add(index / 128 - scales.len()),
                )
            }
        };
        (byte, f32::from_bits((scale as u32) << 16))
    }

    #[inline(always)]
    fn dot4(word: u8, scale: f32, x: [f32; 4], mut sum: f32) -> f32 {
        let mut i = 0;
        while i < 4 {
            let code = ((word >> (i * 2)) & 3) as i32 - 2;
            sum = (code as f32 * scale).mul_add(x[i], sum);
            i += 1;
        }
        sum
    }

    #[inline(always)]
    fn sum_warp(mut value: f32) -> f32 {
        let mut shift = 16;
        while shift > 0 {
            value += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, value, shift);
            shift /= 2;
        }
        value
    }

    #[kernel]
    pub unsafe fn route(logits: &[f32], selected: *mut u32, probabilities: *mut f32) {
        let token = thread::index_1d().get();
        if token < logits.len() / 256 {
            let mut scores = [f32::NEG_INFINITY; 8];
            let mut indices = [u32::MAX; 8];
            let mut expert = 0;
            while expert < 256 {
                let score = logits[token * 256 + expert];
                if score > scores[7] {
                    let mut slot = 7;
                    while slot > 0 && score > scores[slot - 1] {
                        scores[slot] = scores[slot - 1];
                        indices[slot] = indices[slot - 1];
                        slot -= 1;
                    }
                    scores[slot] = score;
                    indices[slot] = expert as u32;
                }
                expert += 1;
            }
            let maximum = scores[0];
            let mut sum = 0.0;
            let mut i = 0;
            while i < 8 {
                scores[i] = (scores[i] - maximum).exp();
                sum += scores[i];
                i += 1;
            }
            i = 0;
            while i < 8 {
                unsafe {
                    *selected.add(token * 8 + i) = indices[i];
                    *probabilities.add(token * 8 + i) = scores[i] / sum;
                }
                i += 1;
            }
        }
    }

    #[kernel]
    pub unsafe fn gate_up<const HOST: bool>(
        hidden: usize,
        intermediate: usize,
        selected: &[u32],
        input: &[f32],
        gate: &[u8],
        gate_scales: &[u16],
        up: &[u8],
        up_scales: &[u16],
        host_gate: *const u8,
        host_gate_scales: *const u16,
        host_up: *const u8,
        host_up_scales: *const u16,
        output: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let row = tid / 32;
        let lane = tid % 32;
        if row < input.len() / hidden * 8 * intermediate {
            let choice = row / intermediate;
            let token = choice / 8;
            if selected[choice] >= 256 {
                if lane == 0 {
                    unsafe {
                        *output.add(row) = f32::NAN;
                    }
                }
                return;
            }
            let expert_row = selected[choice] as usize * intermediate + row % intermediate;
            let mut g = 0.0;
            let mut u = 0.0;
            let mut column = lane * 4;
            while column < hidden {
                // SAFETY: validated dimensions are multiples of128. Every
                // row and four-value offset preserves allocation alignment.
                let x = unsafe {
                    input
                        .as_ptr()
                        .add(token * hidden + column)
                        .cast::<Float4>()
                        .read()
                        .values
                };
                let index = expert_row * hidden + column;
                let (word, scale) =
                    coefficients::<HOST>(gate, gate_scales, host_gate, host_gate_scales, index);
                g = dot4(word, scale, x, g);
                let (word, scale) =
                    coefficients::<HOST>(up, up_scales, host_up, host_up_scales, index);
                u = dot4(word, scale, x, u);
                column += 128;
            }
            g = sum_warp(g);
            u = sum_warp(u);
            if lane == 0 {
                unsafe {
                    *output.add(row) = (g / (1.0 + (-g).exp())) * u;
                }
            }
        }
    }

    #[kernel]
    pub unsafe fn down<const HOST: bool>(
        hidden: usize,
        intermediate: usize,
        selected: &[u32],
        probabilities: &[f32],
        activated: &[f32],
        weights: &[u8],
        scales: &[u16],
        host_weights: *const u8,
        host_scales: *const u16,
        output: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let row = tid / 32;
        let lane = tid % 32;
        if row < activated.len() / intermediate / 8 * hidden {
            let token = row / hidden;
            let mut result = 0.0;
            let mut choice = token * 8;
            while choice < (token + 1) * 8 {
                if selected[choice] >= 256 {
                    if lane == 0 {
                        unsafe {
                            *output.add(row) = f32::NAN;
                        }
                    }
                    return;
                }
                let expert_row = selected[choice] as usize * hidden + row % hidden;
                let mut sum = 0.0;
                let mut column = lane * 4;
                while column < intermediate {
                    // SAFETY: rows and columns preserve16-byte alignment.
                    let x = unsafe {
                        activated
                            .as_ptr()
                            .add(choice * intermediate + column)
                            .cast::<Float4>()
                            .read()
                            .values
                    };
                    let index = expert_row * intermediate + column;
                    let (word, scale) =
                        coefficients::<HOST>(weights, scales, host_weights, host_scales, index);
                    sum = dot4(word, scale, x, sum);
                    column += 128;
                }
                sum = sum_warp(sum);
                result += sum * probabilities[choice];
                choice += 1;
            }
            if lane == 0 {
                unsafe {
                    *output.add(row) = result;
                }
            }
        }
    }

    #[kernel]
    pub unsafe fn combine(
        hidden: usize,
        routed: &[f32],
        shared: &[f32],
        gate: &[f32],
        output: *mut f32,
    ) {
        let i = thread::index_1d().get();
        if i < routed.len() {
            let scale = 1.0 / (1.0 + (-gate[i / hidden]).exp());
            unsafe {
                *output.add(i) = shared[i].mul_add(scale, routed[i]);
            }
        }
    }
}
