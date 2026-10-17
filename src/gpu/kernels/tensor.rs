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
}
