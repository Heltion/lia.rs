//! Scalar storage decoding used as a correctness reference for GPU kernels.
use crate::codebooks::*;
use anyhow::anyhow;

#[inline(always)]
fn u16_at(b: &[u8], i: usize) -> u16 {
    assert!(i <= b.len() && b.len() - i >= 2);
    let p = b.as_ptr().wrapping_add(i);
    if (p as usize).is_multiple_of(2) {
        // SAFETY: bounds and alignment are checked; the immutable slice owns
        // both bytes for this read. GGUF words use little-endian storage.
        unsafe { u16::from_le(p.cast::<u16>().read()) }
    } else {
        b[i] as u16 | ((b[i + 1] as u16) << 8)
    }
}
#[inline(always)]
fn u32_at(b: &[u8], i: usize) -> u32 {
    assert!(i <= b.len() && b.len() - i >= 4);
    let p = b.as_ptr().wrapping_add(i);
    if (p as usize).is_multiple_of(4) {
        // SAFETY: checked four-byte extent and alignment within a live slice.
        unsafe { u32::from_le(p.cast::<u32>().read()) }
    } else {
        u16_at(b, i) as u32 | ((u16_at(b, i + 2) as u32) << 16)
    }
}
/// IEEE binary16 conversion, including subnormals, infinities and NaNs.
#[inline(always)]
pub fn half(bits: u16) -> f32 {
    f16::from_bits(bits) as f32
}

#[cfg(test)]
fn half_reference(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exp = (bits >> 10) & 31;
    let frac = (bits & 1023) as u32;
    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign);
        }
        let shift = frac.leading_zeros() - 21;
        return f32::from_bits(sign | ((113 - shift) << 23) | ((frac << shift) & 1023) << 13);
    }
    f32::from_bits(sign | (if exp == 31 { 255 } else { exp as u32 + 112 }) << 23 | frac << 13)
}
#[inline(always)]
fn h(b: &[u8], i: usize) -> f32 {
    half(u16_at(b, i))
}
#[inline(always)]
fn parity_sign(bits: u32, lane: usize) -> f32 {
    let signs = bits | ((bits.count_ones() & 1) << 7);
    if signs & (1 << lane) == 0 { 1.0 } else { -1.0 }
}
#[inline(always)]
fn sign(bits: u8, lane: usize) -> f32 {
    if bits & (1 << lane) == 0 { 1.0 } else { -1.0 }
}
#[inline(always)]
fn byte(word: u64, lane: usize) -> f32 {
    ((word >> (lane * 8)) & 255) as f32
}
#[inline(always)]
fn signed_byte(word: u64, lane: usize) -> f32 {
    ((word >> (lane * 8)) as u8 as i8) as f32
}

/// Decode exactly one block. Only formats required by the target model are accepted.
pub fn decode_block(kind: u32, b: &[u8]) -> anyhow::Result<Vec<f32>> {
    let (count, bytes) = match kind {
        0 => (1, 4),
        30 => (1, 2),
        10 => (256, 84),
        12 => (256, 144),
        14 => (256, 210),
        16 => (256, 66),
        17 => (256, 74),
        18 => (256, 98),
        19 => (256, 50),
        21 => (256, 110),
        22 => (256, 82),
        23 => (256, 136),
        29 => (256, 56),
        _ => return Err(anyhow!("unimplemented storage format {kind}")),
    };
    if b.len() != bytes {
        return Err(anyhow!(
            "format {kind}: expected {bytes} bytes, got {}",
            b.len()
        ));
    }
    if kind == 0 {
        return Ok(vec![f32::from_bits(u32_at(b, 0))]);
    }
    if kind == 30 {
        return Ok(vec![f32::from_bits((u16_at(b, 0) as u32) << 16)]);
    }
    let mut out = vec![0.; count];
    for (i, y) in out.iter_mut().enumerate() {
        *y = decode_value(kind, b, i);
    }
    Ok(out)
}

/// Caller validates the storage type, block extent and element index.
#[inline(always)]
pub fn decode_value(kind: u32, b: &[u8], i: usize) -> f32 {
    if kind == 0 {
        return f32::from_bits(u32_at(b, 0));
    }
    if kind == 30 {
        return f32::from_bits((u16_at(b, 0) as u32) << 16);
    }
    let g = i / 32;
    let l = i % 32;
    let oct = i / 8;
    let lane = i % 8;
    match kind {
        10 => {
            let sc = b[i / 16];
            let q = (b[16 + (i / 128) * 32 + i % 32] >> (2 * ((i % 128) / 32))) & 3;
            h(b, 80) * (sc & 15) as f32 * q as f32 - h(b, 82) * (sc >> 4) as f32
        }
        12 => {
            let s = &b[4..16];
            let sc = if g < 4 {
                s[g] & 63
            } else {
                (s[g + 4] & 15) | ((s[g - 4] >> 6) << 4)
            };
            let min = if g < 4 {
                s[g + 4] & 63
            } else {
                (s[g + 4] >> 4) | ((s[g] >> 6) << 4)
            };
            let q = (b[16 + (g / 2) * 32 + l] >> (4 * (g % 2))) & 15;
            h(b, 0) * sc as f32 * q as f32 - h(b, 2) * min as f32
        }
        14 => {
            let group = (i % 128) / 32;
            let ql = b[(i / 128) * 64 + (group % 2) * 32 + l];
            let qh = b[128 + (i / 128) * 32 + l];
            let q = ((ql >> (4 * (group / 2))) & 15) | (((qh >> (2 * group)) & 3) << 4);
            h(b, 208) * (b[192 + i / 16] as i8) as f32 * (q as i32 - 32) as f32
        }
        16 => {
            let aux = u32_at(b, 2 + g * 8 + 4);
            let grid = IQ2XXS_GRID[b[2 + g * 8 + l / 8] as usize];
            h(b, 0)
                * (0.5 + (aux >> 28) as f32)
                * 0.25
                * byte(grid, lane)
                * parity_sign((aux >> (7 * (l / 8))) & 127, lane)
        }
        17 => {
            let q = u16_at(b, 2 + oct * 2);
            let sc = (b[66 + g] >> (4 * (l / 16))) & 15;
            h(b, 0)
                * (0.5 + sc as f32)
                * 0.25
                * byte(IQ2XS_GRID[(q & 511) as usize], lane)
                * parity_sign((q >> 9) as u32, lane)
        }
        18 => {
            let aux = u32_at(b, 66 + 4 * g);
            let grid = IQ3XXS_GRID[b[2 + i / 4] as usize];
            h(b, 0)
                * (0.5 + (aux >> 28) as f32)
                * 0.5
                * byte(grid as u64, i % 4)
                * parity_sign((aux >> (7 * (l / 8))) & 127, lane)
        }
        19 => {
            let qh = u16_at(b, 34 + 2 * g);
            let index = b[2 + oct] as usize | ((((qh >> (3 * (l / 8))) & 7) as usize) << 8);
            let delta = if qh & 0x8000 == 0 { 0.125 } else { -0.125 };
            h(b, 0)
                * (2 * ((qh >> 12) & 7) + 1) as f32
                * (signed_byte(IQ1S_GRID[index], lane) + delta)
        }
        21 => {
            let nib = i / 4;
            let high = ((b[66 + g] >> (nib % 8)) & 1) as usize;
            let grid = IQ3S_GRID[b[2 + nib] as usize | (high << 8)];
            let sc = (b[106 + g / 2] >> (4 * (g % 2))) & 15;
            h(b, 0) * (1 + 2 * sc) as f32 * byte(grid as u64, i % 4) * sign(b[74 + oct], lane)
        }
        22 => {
            let high = ((b[66 + g] >> (2 * (l / 8))) & 3) as usize;
            let index = b[2 + oct] as usize | (high << 8);
            let sc = (b[74 + g] >> (4 * (l / 16))) & 15;
            h(b, 0)
                * (0.5 + sc as f32)
                * 0.25
                * byte(IQ2S_GRID[index], lane)
                * sign(b[34 + oct], lane)
        }
        23 => {
            let sc = ((b[4 + g / 2] >> (4 * (g % 2))) & 15) as u16
                | (((u16_at(b, 2) >> (2 * g)) & 3) << 4);
            let q = (b[8 + g * 16 + l % 16] >> (4 * (l / 16))) & 15;
            h(b, 0) * (sc as i32 - 32) as f32 * KVALUES_IQ4NL[q as usize] as f32
        }
        29 => {
            let bits = (u16_at(b, 48) >> 12)
                | ((u16_at(b, 50) >> 8) & 0xf0)
                | ((u16_at(b, 52) >> 4) & 0xf00)
                | (u16_at(b, 54) & 0xf000);
            let scales = u16_at(b, 48 + 2 * (g / 2));
            let sc = (scales >> (6 * (g % 2) + 3 * (l / 16))) & 7;
            let qh = (b[32 + oct / 2] >> (4 * (oct % 2))) & 15;
            let index = b[oct] as usize | (((qh & 7) as usize) << 8);
            let delta = if qh & 8 == 0 { 0.125 } else { -0.125 };
            half(bits) * (2 * sc + 1) as f32 * (signed_byte(IQ1S_GRID[index], lane) + delta)
        }
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_half_encodings() {
        for bits in 0..=u16::MAX {
            let expected = half_reference(bits);
            let actual = half(bits);
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits(), "half bits {bits}");
            }
        }
    }
    #[test]
    fn half_edges() {
        assert_eq!(half(0), 0.0);
        assert_eq!(half(0x8000).to_bits(), (-0.0f32).to_bits());
        assert_eq!(half(1), 2f32.powi(-24));
        assert_eq!(half(0x3c00), 1.0);
        assert_eq!(half(0x7bff), 65504.0);
        assert_eq!(half(0x7c00), f32::INFINITY);
        assert!(half(0x7e00).is_nan());
    }
}

#[cfg(any(feature = "cuda", test))]
mod packed;
#[cfg(test)]
pub use packed::packed_eight;
#[cfg(feature = "cuda")]
pub use packed::packed_eight_aligned;

#[cfg(any(feature = "cuda", test))]
pub(crate) mod fp8;
