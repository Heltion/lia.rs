//! FP8 weight conversion into 64-row by 64-column tiles.
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[inline(always)]
pub(super) fn packed(word: u32, scale: f32, offset: f32, normalizer: f32) -> u32 {
    let a = (((word as u8 as i8) as f32) * scale + offset) * normalizer;
    let b = ((((word >> 8) as u8 as i8) as f32) * scale + offset) * normalizer;
    let c = ((((word >> 16) as u8 as i8) as f32) * scale + offset) * normalizer;
    let d = ((((word >> 24) as u8 as i8) as f32) * scale + offset) * normalizer;
    cuda_device::convert::cvt_rn_satfinite_e4m3x2_f32(a, b) as u32
        | ((cuda_device::convert::cvt_rn_satfinite_e4m3x2_f32(c, d) as u32) << 16)
}

#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub unsafe fn fp8_pack_weight_tiles<const KIND: u32, const BYTES: usize>(
        cols: usize,
        rows: usize,
        padded: usize,
        normalizer: f32,
        weights: &[u8],
        output: &mut [u32],
    ) {
        let vec = thread::index_1d().get() as usize;
        if vec >= padded / 8 {
            return;
        }
        let row = vec / (cols / 8);
        let col = (vec % (cols / 8)) * 8;
        let (lo, hi, scale, offset) = if row < rows {
            let block = (row * cols + col) / 256;
            // SAFETY: host validated complete quantization blocks and launch bounds.
            let data =
                unsafe { core::slice::from_raw_parts(weights.as_ptr().add(block * BYTES), BYTES) };
            unsafe { crate::quant::packed_eight_aligned(KIND, data, col % 256) }
        } else {
            (0, 0, 0.0, 0.0)
        };
        let at = ((row / 64 * (cols / 64) + col / 64) * 4096 + row % 64 * 64 + col % 64) / 4;
        output[at] = packed(lo, scale, offset, normalizer);
        output[at + 1] = packed(hi, scale, offset, normalizer);
    }
}
