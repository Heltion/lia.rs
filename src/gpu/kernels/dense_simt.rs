//! Float32 SIMT products shared by acoustic component networks.
use cuda_device::{kernel, thread, warp};
use cuda_host::cuda_module;
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub unsafe fn dense_f32(
        columns: usize,
        padded: usize,
        rows: usize,
        input: &[f32],
        weight: &[u8],
        output: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        if group < input.len() / columns * rows {
            let row = group % rows;
            let token = group / rows;
            let mut sum = 0.0f32;
            let mut col = lane;
            while col < columns {
                let w = unsafe {
                    weight
                        .as_ptr()
                        .add((row * padded + col) * 4)
                        .cast::<f32>()
                        .read()
                };
                sum = w.mul_add(input[token * columns + col], sum);
                col += 32;
            }
            let mut delta = 16;
            while delta > 0 {
                sum += warp::shuffle_down_f32_sync(0xffff_ffff, sum, delta);
                delta /= 2;
            }
            if lane == 0 {
                unsafe {
                    *output.add(group) = sum;
                }
            }
        }
    }
}
