//! Exact FP16 W2 operands, converting the two nonzero magnitudes once per scale.
use cuda_device::convert;

pub(super) struct Scale {
    negative: u32,
    positive: u32,
}

impl Scale {
    #[inline(always)]
    pub(super) fn new(bits: u32) -> Self {
        let value = f32::from_bits(bits << 16);
        // Convert both magnitudes independently: doubling an already-rounded
        // FP16 subnormal would not preserve the original operand.
        let pair = convert::cvt_f16x2_f32(value, value * 2.0);
        Self {
            negative: ((pair >> 16) ^ 0x8000) | ((pair ^ 0x8000) << 16),
            positive: (bits & 0x8000) | (pair << 16),
        }
    }

    #[inline(always)]
    pub(super) fn select(&self, codes: u32) -> u32 {
        cuda_device::prmt::prmt(
            self.negative,
            self.positive,
            0x1010 + (codes & 3) * 0x22 + ((codes >> 2) & 3) * 0x2200,
        )
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn fragment(
    weights: &[u8],
    scales: &[u16],
    rows: usize,
    columns: usize,
    expert: usize,
    tile: usize,
    k: usize,
    lane: usize,
) -> [u32; 4] {
    let tile = expert * (rows / 16) + tile;
    let i = (tile * (columns / 16) + k / 16) * 32 + lane;
    // SAFETY: the register layout validates complete aligned weight tiles.
    let word = unsafe { weights.as_ptr().cast::<u16>().add(i).read() } as u32;
    let at = (tile * (columns / 128) + k / 128) * 16 + lane / 4;
    let s = Scale::new(scales[at] as u32);
    let t = Scale::new(scales[at + 8] as u32);
    [
        s.select(word),
        t.select(word >> 4),
        s.select(word >> 8),
        t.select(word >> 12),
    ]
}

#[cfg(test)]
#[cuda_host::cuda_module]
pub(crate) mod device {
    use super::*;

    #[cuda_device::kernel]
    pub unsafe fn exhaustive(output: *mut u32) {
        let i = cuda_device::thread::index_1d().get();
        if i < 65536 * 16 {
            let scale = (i / 16) as u32;
            let exponent = (scale >> 7) & 255;
            let codes = (i % 16) as u32;
            let (new, old) = if (1..=253).contains(&exponent) {
                let bf16 = super::super::moe_register_math::coefficients(codes, scale);
                let (a, b) = convert::cvt_f32x2_bf16x2(bf16);
                (
                    Scale::new(scale).select(codes),
                    convert::cvt_f16x2_f32(a, b),
                )
            } else {
                (0, 0)
            };
            unsafe {
                *output.add(i * 2) = new;
                *output.add(i * 2 + 1) = old;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::gpu::kernels;
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};

    #[test]
    fn half_codes_match_every_normal_bf16_scale_and_code_pair() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        let count = 65536 * 16;
        let sentinel = 0xdead_beefu32;
        let out = DeviceBuffer::from_host(&stream, &vec![sentinel; count * 2 + 17])?;
        // SAFETY: one writer per code pair, followed by an untouched guard.
        unsafe {
            module.moe_half.exhaustive(
                &stream,
                LaunchConfig::for_num_elems(count as u32),
                out.cu_deviceptr() as *mut u32,
            )?;
        }
        let values = out.to_host_vec(&stream)?;
        for scale in 0..65536u32 {
            let exponent = (scale >> 7) & 255;
            if !(1..=253).contains(&exponent) {
                continue;
            }
            let s = f32::from_bits(scale << 16);
            for codes in 0..16u32 {
                let a = ((((codes & 3) as i32 - 2) as f32 * s) as f16).to_bits();
                let b = (((((codes >> 2) & 3) as i32 - 2) as f32 * s) as f16).to_bits();
                let expected = a as u32 | ((b as u32) << 16);
                let at = (scale as usize * 16 + codes as usize) * 2;
                assert_eq!(values[at], expected, "new scale={scale:04x} codes={codes}");
                assert_eq!(
                    values[at + 1],
                    expected,
                    "old scale={scale:04x} codes={codes}"
                );
            }
        }
        assert!(values[count * 2..].iter().all(|&x| x == sentinel));
        Ok(())
    }
}
