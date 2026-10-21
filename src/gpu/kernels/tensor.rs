use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[kernel]
    pub unsafe fn pack_input(input: &[f32], output: *mut u32) {
        let i = thread::index_1d().get();
        if i < input.len() / 2 {
            // SAFETY: the host allocates one packed word per complete pair.
            unsafe {
                *output.add(i) =
                    cuda_device::convert::cvt_f16x2_f32(input[2 * i], input[2 * i + 1]);
            }
        }
    }
    /// Two contiguous planes hold the nearest FP16 value and its FP16 residual.
    #[kernel]
    pub unsafe fn pack_input_residual(input: &[f32], output: *mut u32) {
        let i = thread::index_1d().get();
        if i < input.len() / 2 {
            let a = input[2 * i];
            let b = input[2 * i + 1];
            let high = cuda_device::convert::cvt_f16x2_f32(a, b);
            let (x, y) = cuda_device::convert::cvt_f32x2_f16x2(high);
            let low = cuda_device::convert::cvt_f16x2_f32(a - x, b - y);
            // SAFETY: the host reserves two words per complete input pair.
            unsafe {
                *output.add(i) = high;
                *output.add(input.len() / 2 + i) = low;
            }
        }
    }
}
