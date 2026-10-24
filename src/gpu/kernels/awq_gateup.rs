//! Paired eight-row gate/up fragments activate entirely within each MMA lane.
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub unsafe fn gateup_wide(
        cols: usize,
        rows: usize,
        group: usize,
        batch: usize,
        weights: &[u32],
        zeros: &[u32],
        scales: &[f32],
        input: &[f32],
        output: *mut f32,
    ) {
        static mut TILE: cuda_device::SharedArray<u32, 4608, 16> = cuda_device::SharedArray::UNINIT;
        unsafe {
            super::super::awq_tile::device::compute::<64, 64, 256, false, true>(
                cols,
                rows,
                group,
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
    #[kernel]
    pub unsafe fn gateup_medium(
        cols: usize,
        rows: usize,
        group: usize,
        batch: usize,
        weights: &[u32],
        zeros: &[u32],
        scales: &[f32],
        input: &[f32],
        output: *mut f32,
    ) {
        static mut TILE: cuda_device::SharedArray<u32, 2880, 16> = cuda_device::SharedArray::UNINIT;
        unsafe {
            super::super::awq_tile::device::compute::<48, 32, 192, false, true>(
                cols,
                rows,
                group,
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
    #[kernel]
    pub unsafe fn gateup_narrow(
        cols: usize,
        rows: usize,
        group: usize,
        batch: usize,
        weights: &[u32],
        zeros: &[u32],
        scales: &[f32],
        input: &[f32],
        output: *mut f32,
    ) {
        static mut TILE: cuda_device::SharedArray<u32, 2304, 16> = cuda_device::SharedArray::UNINIT;
        unsafe {
            super::super::awq_tile::device::compute::<32, 32, 128, false, true>(
                cols,
                rows,
                group,
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
    #[inline(always)]
    pub(crate) unsafe fn finish(
        rows: u32,
        batch: u32,
        row: u32,
        token: u32,
        sum: [f32; 4],
        output: *mut f32,
    ) {
        if row < rows {
            if token < batch {
                unsafe {
                    *output.add((token * rows + row) as usize) =
                        sum[2] * sum[0] / (1.0 + (-sum[0]).exp());
                }
            }
            if token + 1 < batch {
                unsafe {
                    *output.add(((token + 1) * rows + row) as usize) =
                        sum[3] * sum[1] / (1.0 + (-sum[1]).exp());
                }
            }
        }
    }
}
