//! Device-resident dynamic normalization and FP8 packing.
use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[inline(always)]
    unsafe fn block_max(mut value: f32, local: usize, shared: *mut f32) -> f32 {
        let lane = local % 32;
        let group = local / 32;
        let mut delta = 16;
        while delta > 0 {
            value = value.max(warp::shuffle_down_f32_sync(0xffff_ffff, value, delta));
            delta /= 2;
        }
        unsafe {
            if lane == 0 {
                *shared.add(group) = value;
            }
            thread::sync_threads();
            if group == 0 {
                value = if lane < 8 { *shared.add(lane) } else { 0.0 };
                delta = 16;
                while delta > 0 {
                    value = value.max(warp::shuffle_down_f32_sync(0xffff_ffff, value, delta));
                    delta /= 2;
                }
                if lane == 0 {
                    *shared = value;
                }
            }
            thread::sync_threads();
            *shared
        }
    }

    #[kernel]
    pub unsafe fn fp8_input_maxima(input: &[f32], output: *mut f32) {
        static mut MAX: cuda_device::SharedArray<f32, 8> = cuda_device::SharedArray::UNINIT;
        let tid = thread::index_1d().get();
        let local = tid % 256;
        let block = tid / 256;
        let mut value = 0.0f32;
        let mut i = block * 1024 + local;
        while i < input.len().min((block + 1) * 1024) {
            let x = input[i];
            value = value.max(if x.is_finite() {
                x.abs()
            } else {
                f32::INFINITY
            });
            i += 256;
        }
        unsafe {
            let shared = cuda_device::SharedArray::as_raw_mut_ptr(&raw mut MAX);
            let maximum = block_max(value, local, shared);
            if local == 0 {
                *output.add(block) = maximum;
            }
        }
    }

    #[kernel]
    pub unsafe fn fp8_input_normalizer(count: usize, maxima: &[f32], scales: *mut f32) {
        static mut MAX: cuda_device::SharedArray<f32, 8> = cuda_device::SharedArray::UNINIT;
        let local = thread::index_1d().get();
        let mut value = 0.0f32;
        let mut i = local;
        while i < count {
            value = value.max(maxima[i]);
            i += 256;
        }
        unsafe {
            let shared = cuda_device::SharedArray::as_raw_mut_ptr(&raw mut MAX);
            let maximum = block_max(value, local, shared);
            if local == 0 {
                let mut scale = if maximum == 0.0 {
                    1.0
                } else if !maximum.is_finite() {
                    f32::NAN
                } else {
                    let exponent = ((maximum.to_bits() >> 23) & 255) as i32 - 127;
                    f32::from_bits((((8 - exponent).clamp(-120, 120) + 127) as u32) << 23)
                };
                if maximum * scale > 448.0 {
                    scale *= 0.5;
                }
                *scales = scale;
                *scales.add(1) = 1.0 / scale;
            }
        }
    }

    #[kernel]
    pub fn fp8_pack_scaled(input: &[f32], scales: &[f32], mut output: DisjointSlice<u32>) {
        let index = thread::index_1d();
        let start = index.get() * 4;
        if let Some(out) = output.get_mut(index) {
            let a = if start < input.len() {
                input[start] * scales[0]
            } else {
                0.0
            };
            let b = if start + 1 < input.len() {
                input[start + 1] * scales[0]
            } else {
                0.0
            };
            let c = if start + 2 < input.len() {
                input[start + 2] * scales[0]
            } else {
                0.0
            };
            let d = if start + 3 < input.len() {
                input[start + 3] * scales[0]
            } else {
                0.0
            };
            *out = cuda_device::convert::cvt_rn_satfinite_e4m3x2_f32(a, b) as u32
                | ((cuda_device::convert::cvt_rn_satfinite_e4m3x2_f32(c, d) as u32) << 16);
        }
    }
}
