//! Recover the second FP16 input piece without expanding expert weights.
use cuda_device::convert;

#[repr(C, align(8))]
struct Pair {
    values: [f32; 2],
}

/// The caller supplies an aligned pair and another pair eight columns later,
/// both within one input row. The result follows the MMA B fragment layout.
#[inline(always)]
pub(super) unsafe fn fragment(input: &[f32], offset: usize) -> [u32; 2] {
    let mut result = [0; 2];
    let mut pair = 0;
    while pair < 2 {
        // SAFETY: the caller validates complete16-column matrix fragments.
        let x = unsafe { input.as_ptr().add(offset + pair * 8).cast::<Pair>().read() }.values;
        let high = convert::cvt_f16x2_f32(x[0], x[1]);
        let (a, b) = convert::cvt_f32x2_f16x2(high);
        result[pair] = convert::cvt_f16x2_f32(x[0] - a, x[1] - b);
        pair += 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use crate::gpu::{
        kernels,
        moe::{Experts, Grouped},
    };
    use crate::quant::gsq::Matrix;
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer};

    fn matrix(up: bool) -> Matrix {
        let mut packed = vec![0x88u8; 256 * 128 * 128 / 2];
        for expert in 0..256 {
            for row in 0..128 {
                let column = if up { 127 } else { row };
                let index = (expert * 128 + row) * 128 + column;
                let shift = index % 2 * 4;
                packed[index / 2] = (packed[index / 2] & !(15 << shift)) | (9 << shift);
            }
        }
        Matrix {
            rows: 256 * 128,
            columns: 128,
            bits: 4,
            group_size: 128,
            packed,
            scales: vec![1.0; 256 * 128],
        }
    }

    #[test]
    fn f32_input_is_preserved_through_int4_experts() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        for offload in [0, 128] {
            let experts = Experts::upload_offloaded(
                &stream,
                128,
                128,
                matrix(false),
                matrix(true),
                matrix(false),
                offload,
            )?;
            let work = Grouped::new(&stream, 1025)?;
            for batch in [17usize, 129, 1025] {
                let input: Vec<f32> = (0..batch * 128)
                    .map(|i| {
                        if i % 128 == 127 {
                            1.0
                        } else {
                            0.1234567 + (i % 101) as f32 * 0.007321
                        }
                    })
                    .collect();
                let selected: Vec<u32> = (0..batch * 8)
                    .map(|i| (if offload == 0 { 0 } else { 248 }) + (i % 8) as u32)
                    .collect();
                let x = DeviceBuffer::from_host(&stream, &input)?;
                let ids = DeviceBuffer::from_host(&stream, &selected)?;
                let probabilities = DeviceBuffer::from_host(&stream, &vec![0.125f32; batch * 8])?;
                let mut activated = DeviceBuffer::zeroed(&stream, batch * 8 * 128)?;
                let mut choices = DeviceBuffer::zeroed(&stream, batch * 8 * 128)?;
                let mut output = DeviceBuffer::zeroed(&stream, batch * 128)?;
                experts.forward_grouped(
                    &module,
                    &stream,
                    &work,
                    &x,
                    None,
                    false,
                    &ids,
                    &probabilities,
                    &mut activated,
                    &mut choices,
                    &mut output,
                )?;
                let mut maximum = 0.0f64;
                for (&actual, &value) in output.to_host_vec(&stream)?.iter().zip(&input) {
                    let value = f64::from(value);
                    let expected = value / (1.0 + (-value).exp());
                    maximum = maximum.max((f64::from(actual) - expected).abs());
                }
                assert!(
                    maximum < 5e-7,
                    "INT4 input residual: batch={batch}, offload={offload}, error={maximum}"
                );
            }
        }
        Ok(())
    }
}
