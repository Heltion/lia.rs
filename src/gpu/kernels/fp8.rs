//! FP8 conversion and native fragment validation before inference integration.
use cuda_device::{DisjointSlice, kernel, thread, wmma};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[kernel]
    pub fn pack(input: &[f32], mut output: DisjointSlice<u16>) {
        let index = thread::index_1d();
        let i = index.get() * 2;
        if let Some(value) = output.get_mut(index) {
            *value = cuda_device::convert::cvt_rn_satfinite_e4m3x2_f32(input[i], input[i + 1]);
        }
    }

    #[kernel]
    pub unsafe fn multiply(a: &[u32], b: &[u32], output: *mut f32) {
        let lane = thread::index_1d().get();
        let row = lane / 4;
        let word = lane % 4;
        // SAFETY: one complete warp; A is16x32, B is8x32 packed bytes;
        // each lane owns two columns in each of two rows of the16x8 result.
        unsafe {
            let af = [
                a[row * 8 + word],
                a[(row + 8) * 8 + word],
                a[row * 8 + word + 4],
                a[(row + 8) * 8 + word + 4],
            ];
            let bf = [b[row * 8 + word], b[row * 8 + word + 4]];
            let result = wmma::mma_m16n8k32_fp8_f32_e4m3_e4m3([0.0; 4], af, bf);
            let column = lane % 4 * 2;
            *output.add(row * 8 + column) = result[0];
            *output.add(row * 8 + column + 1) = result[1];
            *output.add((row + 8) * 8 + column) = result[2];
            *output.add((row + 8) * 8 + column + 1) = result[3];
        }
    }
}

#[cfg(test)]
pub(crate) mod reference {
    // Independent scalar finite E4M3 reference, using its mathematical values.
    pub(crate) fn decode(bits: u8) -> f64 {
        let exponent = (bits >> 3) & 15;
        let fraction = bits & 7;
        let value = if exponent == 15 && fraction == 7 {
            f64::NAN
        } else if exponent == 0 {
            fraction as f64 / 512.0
        } else {
            (1.0 + fraction as f64 / 8.0) * 2.0f64.powi(exponent as i32 - 7)
        };
        if bits & 128 != 0 { -value } else { value }
    }

    pub(crate) fn encode(value: f32) -> u8 {
        if value.is_nan() {
            return 127;
        }
        let sign = if value.is_sign_negative() { 128 } else { 0 };
        let magnitude = value.abs() as f64;
        if magnitude >= 448.0 {
            return sign | 126;
        }
        let (mut low, mut high) = (0u8, 126u8);
        while low < high {
            let middle = low + (high - low) / 2;
            if decode(middle) < magnitude {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        let lower = low.saturating_sub(1);
        let upper_distance = (decode(low) - magnitude).abs();
        let lower_distance = (decode(lower) - magnitude).abs();
        let best = if upper_distance < lower_distance
            || (upper_distance == lower_distance && low & 1 == 0)
        {
            low
        } else {
            lower
        };
        sign | best
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};

    use super::reference::{decode, encode};

    #[test]
    fn reference_roundtrips_and_ties_to_even() {
        for bits in 0..=255 {
            let value = decode(bits);
            if value.is_nan() {
                continue;
            }
            assert_eq!(encode(value as f32), bits);
        }
        for bits in 0..126 {
            let middle = ((decode(bits) + decode(bits + 1)) * 0.5) as f32;
            assert_eq!(encode(middle), if bits & 1 == 0 { bits } else { bits + 1 });
            assert_eq!(encode(middle.next_down()), bits);
            assert_eq!(encode(middle.next_up()), bits + 1);
        }
    }

    #[test]
    fn native_conversion_matches_reference_and_mma_matches_f64() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = crate::gpu::kernels::load(&context)?;
        let mut values = vec![
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
            f32::MAX,
            -f32::MAX,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
        ];
        for bits in 0..=255 {
            values.push(decode(bits) as f32);
        }
        for bits in 0..126 {
            let middle = ((decode(bits) + decode(bits + 1)) * 0.5) as f32;
            for value in [middle.next_down(), middle, middle.next_up()] {
                values.extend([value, -value]);
            }
        }
        if !values.len().is_multiple_of(2) {
            values.push(0.0);
        }
        let input = DeviceBuffer::from_host(&stream, &values)?;
        let mut packed = DeviceBuffer::<u16>::zeroed(&stream, values.len() / 2)?;
        // SAFETY: complete pairs; every output thread has two readable inputs.
        unsafe {
            module.fp8.pack(
                &stream,
                LaunchConfig::for_num_elems(packed.len() as u32),
                &input,
                &mut packed,
            )?;
        }
        for (pair, expected) in packed
            .to_host_vec(&stream)?
            .iter()
            .zip(values.as_chunks::<2>().0)
        {
            for (bits, value) in pair.to_le_bytes().into_iter().zip(expected) {
                if value.is_nan() {
                    assert!(decode(bits).is_nan());
                } else {
                    assert_eq!(bits, encode(*value), "conversion of {value}");
                }
            }
        }
        let make = |length: usize, step: usize| -> Vec<u8> {
            (0..length)
                .map(|i| {
                    let bits = ((i * step + 3) % 256) as u8;
                    if decode(bits).is_nan() { 0 } else { bits }
                })
                .collect()
        };
        let a = make(16 * 32, 17);
        let b = make(8 * 32, 29);
        let words = |bytes: &[u8]| -> Vec<u32> {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect()
        };
        let da = DeviceBuffer::from_host(&stream, &words(&a))?;
        let db = DeviceBuffer::from_host(&stream, &words(&b))?;
        let output = DeviceBuffer::<f32>::zeroed(&stream, 16 * 8)?;
        // SAFETY: exactly32 lanes and complete matrices/output, with disjoint stores.
        unsafe {
            module.fp8.multiply(
                &stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                &da,
                &db,
                output.cu_deviceptr() as *mut f32,
            )?;
        }
        let actual = output.to_host_vec(&stream)?;
        for row in 0..16 {
            for column in 0..8 {
                let (mut expected, mut magnitude) = (0.0f64, 0.0f64);
                for k in 0..32 {
                    let product = decode(a[row * 32 + k]) * decode(b[column * 32 + k]);
                    expected += product;
                    magnitude += product.abs();
                }
                let value = actual[row * 8 + column] as f64;
                anyhow::ensure!(
                    value.is_finite() && (value - expected).abs() <= 3e-6 * magnitude.max(1.0),
                    "FP8 MMA row={row} column={column}: {value} vs {expected}"
                );
            }
        }
        Ok(())
    }
}
