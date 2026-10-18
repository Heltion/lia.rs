//! Conservative metadata bounds for experimental FP8 tensor normalization.
use super::{half, u16_at};
use crate::codebooks::*;
use anyhow::{Result, anyhow, ensure};

fn grid_max(words: impl Iterator<Item = u64>) -> f64 {
    words
        .flat_map(u64::to_le_bytes)
        .map(|b| (b as i8 as f64).abs())
        .fold(0.0, f64::max)
}

fn coefficient_bound(kind: u32) -> Result<f64> {
    Ok(match kind {
        10 | 12 | 14 => 0.0,
        16 => grid_max(IQ2XXS_GRID.iter().copied()) * 3.875,
        17 => grid_max(IQ2XS_GRID.iter().copied()) * 3.875,
        18 => grid_max(IQ3XXS_GRID.iter().map(|&v| v as u64)) * 7.75,
        19 | 29 => grid_max(IQ1S_GRID.iter().copied()) * 16.875,
        21 => grid_max(IQ3S_GRID.iter().map(|&v| v as u64)) * 31.0,
        22 => grid_max(IQ2S_GRID.iter().copied()) * 3.875,
        23 => {
            KVALUES_IQ4NL
                .iter()
                .map(|&x| (x as f64).abs())
                .fold(0.0, f64::max)
                * 32.0
        }
        _ => return Err(anyhow!("unsupported FP8 source format {kind}")),
    })
}

fn block_bound(kind: u32, bytes: &[u8], coefficient: f64) -> Result<f64> {
    let h = |at| (half(u16_at(bytes, at)) as f64).abs();
    let bound = match kind {
        10 => 45.0 * h(80) + 15.0 * h(82),
        12 => 945.0 * h(0) + 63.0 * h(2),
        14 => 4096.0 * h(208),
        29 => {
            let bits = (u16_at(bytes, 48) >> 12)
                | ((u16_at(bytes, 50) >> 8) & 0xf0)
                | ((u16_at(bytes, 52) >> 4) & 0xf00)
                | (u16_at(bytes, 54) & 0xf000);
            (half(bits) as f64).abs() * coefficient
        }
        _ => h(0) * coefficient,
    };
    ensure!(
        bound.is_finite(),
        "nonfinite FP8 source metadata for format {kind}"
    );
    Ok(bound)
}

pub(crate) fn normalizer(kind: u32, bytes: &[u8]) -> Result<f32> {
    let coefficient = coefficient_bound(kind)?;
    let (_, size) = crate::gguf::layout(kind).ok_or_else(|| anyhow!("unknown weight layout"))?;
    let size = size as usize;
    ensure!(
        !bytes.is_empty() && bytes.len().is_multiple_of(size),
        "incomplete FP8 source blocks"
    );
    let mut bound = 0.0f64;
    for block in bytes.chunks_exact(size) {
        bound = bound.max(block_bound(kind, block, coefficient)?);
    }
    if bound == 0.0 {
        return Ok(1.0);
    }
    let exponent = (448.0 / bound).log2().floor().clamp(-120.0, 120.0) as i32;
    let mut scale = 2.0f32.powi(exponent);
    while bound * scale as f64 > 448.0 {
        scale *= 0.5;
    }
    ensure!(
        scale.is_finite() && scale > 0.0,
        "invalid FP8 tensor normalizer"
    );
    Ok(scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_bounds_cover_independently_decoded_blocks() -> Result<()> {
        for kind in [10, 12, 14, 16, 17, 18, 19, 21, 22, 23, 29] {
            let (_, size) = crate::gguf::layout(kind).unwrap();
            let coefficient = coefficient_bound(kind)?;
            for sample in 0..64 {
                let mut bytes: Vec<u8> = (0..size as usize)
                    .map(|i| ((i * 73 + sample * 31 + 19) % 256) as u8)
                    .collect();
                let bits =
                    [0u16, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3800, 0xb800, 0x7bff][sample % 8];
                let offset = match kind {
                    10 => 80,
                    14 => 208,
                    _ => 0,
                };
                if kind == 29 {
                    for j in 0..4 {
                        let at = 48 + j * 2;
                        let word = (u16_at(&bytes, at) & 4095) | (((bits >> (4 * j)) & 15) << 12);
                        bytes[at..at + 2].copy_from_slice(&word.to_le_bytes());
                    }
                } else {
                    bytes[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
                }
                if kind == 10 || kind == 12 {
                    bytes[offset + 2..offset + 4].copy_from_slice(&0xb400u16.to_le_bytes());
                }
                let bound = block_bound(kind, &bytes, coefficient)?;
                let scale = normalizer(kind, &bytes)?;
                assert_eq!(
                    scale.to_bits() & 0x7fffff,
                    0,
                    "normalizer is not a power of two"
                );
                for value in super::super::decode_block(kind, &bytes)? {
                    assert!(
                        (value as f64).abs() <= bound,
                        "format{kind} value{value} bound{bound}"
                    );
                    assert!((value as f64 * scale as f64).abs() <= 448.0);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn invalid_metadata_and_incomplete_blocks_are_rejected() {
        let mut bytes = vec![0; 66];
        assert_eq!(normalizer(16, &bytes).unwrap(), 1.0);
        bytes[..2].copy_from_slice(&0x7c00u16.to_le_bytes());
        assert!(normalizer(16, &bytes).is_err());
        bytes[..2].copy_from_slice(&0x7e00u16.to_le_bytes());
        assert!(normalizer(16, &bytes).is_err());
        assert!(normalizer(16, &bytes[..65]).is_err());
        assert!(normalizer(16, &[]).is_err());
        assert!(normalizer(999, &bytes).is_err());
    }
}
