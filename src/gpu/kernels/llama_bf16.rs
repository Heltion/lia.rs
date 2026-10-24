//! Aligned eight-value loads with independent FP32 accumulators for small batches.
use cuda_device::{kernel, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    struct Words {
        values: [u32; 4],
    }
    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    struct Floats {
        values: [f32; 4],
    }

    #[kernel]
    pub unsafe fn llama_bf16_vector<const BATCH: usize>(
        columns: usize,
        rows: usize,
        batch: usize,
        weights: &[u8],
        input: &[f32],
        output: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let row = tid / 32;
        let lane = tid % 32;
        if row < rows {
            let chunks = if BATCH <= 2 { 2 } else { 1 };
            let mut sums = [[0.0_f32; 8]; BATCH];
            let mut column = lane * 8;
            while column < columns {
                // Independent loads precede arithmetic to overlap memory latency.
                let first = unsafe {
                    weights
                        .as_ptr()
                        .add((row * columns + column) * 2)
                        .cast::<Words>()
                        .read()
                };
                let second = if BATCH <= 2 {
                    unsafe {
                        weights
                            .as_ptr()
                            .add((row * columns + column + 256) * 2)
                            .cast::<Words>()
                            .read()
                    }
                } else {
                    first
                };
                let packed = [first, second];
                let mut token = 0;
                #[unroll]
                while token < BATCH {
                    if token < batch {
                        let mut chunk = 0;
                        #[unroll]
                        while chunk < 2 {
                            if BATCH > 2 && chunk == 1 {
                                break;
                            }
                            let at = token * columns + column + chunk * 256;
                            let a = unsafe { input.as_ptr().add(at).cast::<Floats>().read() };
                            let b = unsafe { input.as_ptr().add(at + 4).cast::<Floats>().read() };
                            let x = [
                                a.values[0],
                                a.values[1],
                                a.values[2],
                                a.values[3],
                                b.values[0],
                                b.values[1],
                                b.values[2],
                                b.values[3],
                            ];
                            let mut j = 0;
                            #[unroll]
                            while j < 4 {
                                let low = f32::from_bits(packed[chunk].values[j] << 16);
                                let high = f32::from_bits(packed[chunk].values[j] & 0xffff_0000);
                                let sum = &mut sums[token][chunk * 4 + j];
                                *sum = low.mul_add(x[j * 2], *sum);
                                *sum = high.mul_add(x[j * 2 + 1], *sum);
                                j += 1;
                            }
                            chunk += 1;
                        }
                    }
                    token += 1;
                }
                column += chunks * 256;
            }
            let mut token = 0;
            #[unroll]
            while token < BATCH {
                if token < batch {
                    let values = sums[token];
                    let mut value = ((values[0] + values[1]) + (values[2] + values[3]))
                        + ((values[4] + values[5]) + (values[6] + values[7]));
                    let mut offset = 16;
                    while offset > 0 {
                        value += warp::shuffle_down_f32_sync(0xffff_ffff, value, offset);
                        offset /= 2;
                    }
                    if lane == 0 {
                        unsafe {
                            *output.add(token * rows + row) = value;
                        }
                    }
                }
                token += 1;
            }
        }
    }
    #[kernel]
    pub unsafe fn llama_pack_weights(weights: &[u8], output: *mut u8) {
        let at = thread::index_1d().get() * 16;
        if at < weights.len() {
            let raw = unsafe { weights.as_ptr().add(at).cast::<Words>().read() };
            let convert = |word: u32| {
                cuda_device::convert::cvt_f16x2_f32(
                    f32::from_bits(word << 16),
                    f32::from_bits(word & 0xffff_0000),
                )
            };
            let packed = Words {
                values: [
                    convert(raw.values[0]),
                    convert(raw.values[1]),
                    convert(raw.values[2]),
                    convert(raw.values[3]),
                ],
            };
            unsafe {
                output.add(at).cast::<Words>().write(packed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::gpu::kernels;
    use anyhow::{Result, ensure};
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};

    #[test]
    #[ignore = "requires CUDA"]
    fn prefill_shadows_match_all_bf16_to_f16_patterns() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        let mut bytes = Vec::with_capacity(65536 * 2);
        for word in 0..=u16::MAX {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        let source = DeviceBuffer::from_host(&stream, &bytes)?;
        let packed = DeviceBuffer::<u8>::zeroed(&stream, bytes.len())?;
        // SAFETY: complete aligned eight-value vectors and equal-sized disjoint output.
        unsafe {
            module.llama_bf16.llama_pack_weights(
                &stream,
                LaunchConfig::for_num_elems((bytes.len() / 16) as u32),
                &source,
                packed.cu_deviceptr() as *mut u8,
            )?;
        }
        let actual = packed.to_host_vec(&stream)?;
        for word in 0..=u16::MAX {
            let at = word as usize * 2;
            let value = u16::from_le_bytes([actual[at], actual[at + 1]]);
            let expected = (f32::from_bits(u32::from(word) << 16) as f16).to_bits();
            if word & 0x7f80 == 0x7f80 && word & 0x007f != 0 {
                ensure!(
                    value & 0x7c00 == 0x7c00 && value & 0x03ff != 0,
                    "BF16 NaN became a non-NaN half"
                );
            } else {
                ensure!(
                    value == expected,
                    "BF16 shadow {word:04x}: {value:04x} vs {expected:04x}"
                );
            }
        }
        println!("All 65,536 BF16 patterns match native half conversion (NaN payloads ignored)");
        Ok(())
    }
}
