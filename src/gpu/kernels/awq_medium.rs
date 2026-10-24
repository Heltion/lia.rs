//! Six-warp AWQ tile avoids a second weight decode for 33..48 input tokens.
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub unsafe fn awq_fp16_medium(
        cols: usize,
        rows: usize,
        group_size: usize,
        batch: usize,
        weights: &[u32],
        zeros: &[u32],
        scales: &[f32],
        input: &[f32],
        output: *mut f32,
    ) {
        static mut TILE: cuda_device::SharedArray<u32, 2880, 16> = cuda_device::SharedArray::UNINIT;
        unsafe {
            super::super::awq_tile::device::compute::<48, 32, 192, false, false>(
                cols,
                rows,
                group_size,
                batch,
                weights,
                zeros,
                scales,
                input,
                output,
                cuda_device::SharedArray::as_raw_mut_ptr(&raw mut TILE),
                thread::index_1d().get() as u32,
                1,
            );
        }
    }
}
