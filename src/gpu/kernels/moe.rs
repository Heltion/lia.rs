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
    fn coefficients<const HOST: bool, const BITS: u32>(
        packed: &[u8],
        scales: &[u16],
        host: *const u8,
        host_scales: *const u16,
        index: usize,
    ) -> (u16, f32) {
        let at = index / (8 / BITS as usize);
        let ptr = if !HOST || at < packed.len() {
            unsafe { packed.as_ptr().add(at) }
        } else {
            unsafe { host.add(at - packed.len()) }
        };
        let word = if BITS == 4 {
            unsafe { ptr.cast::<u16>().read() }
        } else {
            (unsafe { ptr.read() }) as u16
        };
        let scale_at = index / 128;
        let scale = if !HOST || scale_at < scales.len() {
            scales[scale_at]
        } else {
            unsafe { *host_scales.add(scale_at - scales.len()) }
        };
        (word, f32::from_bits((scale as u32) << 16))
    }

    #[inline(always)]
    fn dot4<const BITS: u32>(word: u16, scale: f32, x: [f32; 4], mut sum: f32) -> f32 {
        let mut i = 0;
        while i < 4 {
            let code = ((word as u32 >> (i * BITS)) & ((1 << BITS) - 1)) as i32 - (1 << (BITS - 1));
            sum = (code as f32 * scale).mul_add(x[i as usize], sum);
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
        let tid = thread::index_1d().get();
        let token = tid / 32;
        let lane = tid % 32;
        if token >= logits.len() / 256 {
            return;
        }
        let mut values = [f32::NEG_INFINITY; 8];
        let mut j = 0;
        while j < 8 {
            values[j] = logits[token * 256 + lane + j * 32];
            j += 1;
        }
        super::super::moe_route::select!(values, lane, token, selected, probabilities);
    }

    #[kernel]
    pub unsafe fn gate_up<const HOST: bool, const BITS: u32>(
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
                let (word, scale) = coefficients::<HOST, BITS>(
                    gate,
                    gate_scales,
                    host_gate,
                    host_gate_scales,
                    index,
                );
                g = dot4::<BITS>(word, scale, x, g);
                let (word, scale) =
                    coefficients::<HOST, BITS>(up, up_scales, host_up, host_up_scales, index);
                u = dot4::<BITS>(word, scale, x, u);
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
    pub unsafe fn down<const HOST: bool, const BITS: u32>(
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
        static mut PARTIALS: cuda_device::SharedArray<f32, 8, 16> =
            cuda_device::SharedArray::UNINIT;
        let tid = thread::index_1d().get();
        let row = tid / 256;
        let lane = tid % 32;
        let slot = tid % 256 / 32;
        if row >= activated.len() / intermediate / 8 * hidden {
            return;
        }
        let choice = row / hidden * 8 + slot;
        let mut sum = 0.0;
        if selected[choice] < 256 {
            let expert_row = selected[choice] as usize * hidden + row % hidden;
            let mut column = lane * 4;
            while column < intermediate {
                // SAFETY: validated rows and columns preserve vector alignment.
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
                    coefficients::<HOST, BITS>(weights, scales, host_weights, host_scales, index);
                sum = dot4::<BITS>(word, scale, x, sum);
                column += 128;
            }
        } else {
            sum = f32::NAN;
        }
        sum = sum_warp(sum);
        let partials = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut PARTIALS) };
        if lane == 0 {
            unsafe { *partials.add(slot) = sum };
        }
        // Each CTA owns one output row and eight expert warps. Merge in the
        // original routing order, preserving the serial kernel's sum order.
        thread::sync_threads();
        if tid % 256 == 0 {
            let mut result = 0.0;
            let mut i = 0;
            while i < 8 {
                result += unsafe { *partials.add(i) } * probabilities[row / hidden * 8 + i];
                i += 1;
            }
            unsafe { *output.add(row) = result };
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
