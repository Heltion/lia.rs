//! Scalar storage decoding used as a correctness reference for GPU kernels.
use crate::codebooks::*;

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
pub fn decode_block(kind: u32, b: &[u8]) -> Result<Vec<f32>, String> {
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
        _ => return Err(format!("unimplemented storage format {kind}")),
    };
    if b.len() != bytes {
        return Err(format!(
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

#[inline(always)]
#[cfg(any(feature = "cuda", test))]
fn word(b: &[u8], i: usize) -> u64 {
    assert!(i <= b.len() && b.len() - i >= 8);
    let p = b.as_ptr().wrapping_add(i);
    if (p as usize).is_multiple_of(8) {
        // SAFETY: checked eight-byte extent and alignment in a live immutable slice.
        unsafe { u64::from_le(p.cast::<u64>().read()) }
    } else {
        u32_at(b, i) as u64 | ((u32_at(b, i + 4) as u64) << 32)
    }
}
#[inline(always)]
#[cfg(any(feature = "cuda", test))]
fn signed_grid(grid: u64, signs: u32) -> u64 {
    // Spread a four-bit sign nibble into four independent byte positions.
    // Nonzero codebook magnitudes keep the byte-wise two's-complement add
    // from carrying into the neighbouring byte.
    let low = ((signs & 15) * 0x00204081) & 0x01010101;
    let high = (((signs >> 4) & 15) * 0x00204081) & 0x01010101;
    let lo = ((grid as u32) ^ (low * 255)).wrapping_add(low);
    let hi = (((grid >> 32) as u32) ^ (high * 255)).wrapping_add(high);
    lo as u64 | ((hi as u64) << 32)
}
#[inline(always)]
#[cfg(any(feature = "cuda", test))]
fn even_signs(bits: u32) -> u32 {
    bits | ((bits.count_ones() & 1) << 7)
}

/// Eight integral weight coefficients, one multiplicative scale and one offset.
/// The caller supplies a valid block and an eight-aligned index in 0..256.
/// CPU entry point accepts unaligned block slices.
#[inline(always)]
#[cfg(test)]
pub fn packed_eight(kind: u32, b: &[u8], i: usize) -> (u32, u32, f32, f32) {
    packed_eight_impl::<false>(kind, b, i)
}

#[inline(always)]
#[cfg(feature = "cuda")]
fn iq4_permute(q: u32) -> u32 {
    // Select four bytes from each eight-entry half of the signed nonlinear
    // codebook. PRMT reads registers, avoiding divergent byte-table loads.
    let control = (q & 7) | ((q >> 4) & 0x70) | ((q >> 8) & 0x700) | ((q >> 12) & 0x7000);
    let low = cuda_device::prmt::prmt(0xbfad9881, 0xf6eaddcf, control);
    let high = cuda_device::prmt::prmt(0x26190d01, 0x71594535, control);
    let mask = ((q & 0x08080808) >> 3) * 255;
    (low & !mask) | (high & mask)
}
#[inline(always)]
#[cfg(any(feature = "cuda", test))]
fn iq4_values<const ALIGNED: bool>(q: u64) -> u64 {
    #[cfg(feature = "cuda")]
    if ALIGNED {
        return iq4_permute(q as u32) as u64 | ((iq4_permute((q >> 32) as u32) as u64) << 32);
    }
    let mut values = 0u64;
    let mut j = 0;
    while j < 8 {
        values |= (KVALUES_IQ4NL[((q >> (j * 8)) & 15) as usize] as u8 as u64) << (j * 8);
        j += 1;
    }
    values
}

/// # Safety
/// The complete block must lie in a CUDA allocation whose initial address is
/// 256-byte aligned. The block offset is its original GGUF byte size times
/// its block index; i is eight-aligned and less than 256.
#[inline(always)]
#[cfg(feature = "cuda")]
pub unsafe fn packed_eight_aligned(kind: u32, b: &[u8], i: usize) -> (u32, u32, f32, f32) {
    packed_eight_impl::<true>(kind, b, i)
}

#[inline(always)]
#[cfg(any(feature = "cuda", test))]
fn packed_u16<const ALIGNED: bool>(b: &[u8], i: usize) -> u16 {
    if ALIGNED {
        // SAFETY: every required storage block has even byte size and every
        // multi-byte metadata field used here has an even offset.
        unsafe { b.as_ptr().add(i).cast::<u16>().read() }
    } else {
        u16_at(b, i)
    }
}
#[inline(always)]
#[cfg(any(feature = "cuda", test))]
fn packed_u32<const ALIGNED: bool>(b: &[u8], i: usize) -> u32 {
    if ALIGNED {
        packed_u16::<true>(b, i) as u32 | ((packed_u16::<true>(b, i + 2) as u32) << 16)
    } else {
        u32_at(b, i)
    }
}
#[inline(always)]
#[cfg(any(feature = "cuda", test))]
fn packed_half<const ALIGNED: bool>(b: &[u8], i: usize) -> f32 {
    half(packed_u16::<ALIGNED>(b, i))
}
#[inline(always)]
#[cfg(any(feature = "cuda", test))]
fn packed_word<const ALIGNED: bool>(kind: u32, b: &[u8], i: usize) -> u64 {
    if ALIGNED {
        // SAFETY: IQ4_XS/Q4_K block strides and accessed words are eight-byte
        // aligned; Q2_K is four-byte aligned; Q6_K is two-byte aligned.
        unsafe {
            let p = b.as_ptr().add(i);
            if kind == 12 || kind == 23 {
                p.cast::<u64>().read()
            } else if kind == 10 {
                p.cast::<u32>().read() as u64 | ((p.add(4).cast::<u32>().read() as u64) << 32)
            } else {
                packed_u32::<true>(b, i) as u64 | ((packed_u32::<true>(b, i + 4) as u64) << 32)
            }
        }
    } else {
        word(b, i)
    }
}

#[inline(always)]
#[cfg(any(feature = "cuda", test))]
fn packed_eight_impl<const ALIGNED: bool>(kind: u32, b: &[u8], i: usize) -> (u32, u32, f32, f32) {
    let g = i / 32;
    let l = i % 32;
    let oct = i / 8;
    let (values, scale, offset) = match kind {
        10 => {
            let sc = b[i / 16];
            let shift = 2 * ((i % 128) / 32);
            let values = (packed_word::<ALIGNED>(kind, b, 16 + (i / 128) * 32 + i % 32) >> shift)
                & 0x0303030303030303;
            (
                values,
                packed_half::<ALIGNED>(b, 80) * (sc & 15) as f32,
                -packed_half::<ALIGNED>(b, 82) * (sc >> 4) as f32,
            )
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
            let values = (packed_word::<ALIGNED>(kind, b, 16 + (g / 2) * 32 + l) >> (4 * (g % 2)))
                & 0x0f0f0f0f0f0f0f0f;
            (
                values,
                packed_half::<ALIGNED>(b, 0) * sc as f32,
                -packed_half::<ALIGNED>(b, 2) * min as f32,
            )
        }
        14 => {
            let group = (i % 128) / 32;
            let ql = (packed_word::<ALIGNED>(kind, b, (i / 128) * 64 + (group % 2) * 32 + l)
                >> (4 * (group / 2)))
                & 0x0f0f0f0f0f0f0f0f;
            let qh = (packed_word::<ALIGNED>(kind, b, 128 + (i / 128) * 32 + l) >> (2 * group))
                & 0x0303030303030303;
            let q = (ql | (qh << 4)) ^ 0x2020202020202020;
            let sign = q & 0x2020202020202020;
            (
                q | (sign << 1) | (sign << 2),
                packed_half::<ALIGNED>(b, 208) * (b[192 + i / 16] as i8) as f32,
                0.,
            )
        }
        16 => {
            let aux = packed_u32::<ALIGNED>(b, 2 + g * 8 + 4);
            let grid = IQ2XXS_GRID[b[2 + g * 8 + l / 8] as usize];
            let signs = even_signs((aux >> (7 * (l / 8))) & 127);
            (
                signed_grid(grid, signs),
                packed_half::<ALIGNED>(b, 0) * (0.5 + (aux >> 28) as f32) * 0.25,
                0.,
            )
        }
        17 => {
            let q = packed_u16::<ALIGNED>(b, 2 + oct * 2);
            let sc = (b[66 + g] >> (4 * (l / 16))) & 15;
            (
                signed_grid(IQ2XS_GRID[(q & 511) as usize], even_signs((q >> 9) as u32)),
                packed_half::<ALIGNED>(b, 0) * (0.5 + sc as f32) * 0.25,
                0.,
            )
        }
        18 => {
            let aux = packed_u32::<ALIGNED>(b, 66 + 4 * g);
            let grid = IQ3XXS_GRID[b[2 + oct * 2] as usize] as u64
                | ((IQ3XXS_GRID[b[3 + oct * 2] as usize] as u64) << 32);
            (
                signed_grid(grid, even_signs((aux >> (7 * (l / 8))) & 127)),
                packed_half::<ALIGNED>(b, 0) * (0.5 + (aux >> 28) as f32) * 0.5,
                0.,
            )
        }
        19 => {
            let qh = packed_u16::<ALIGNED>(b, 34 + 2 * g);
            let index = b[2 + oct] as usize | ((((qh >> (3 * (l / 8))) & 7) as usize) << 8);
            let scale = packed_half::<ALIGNED>(b, 0) * (2 * ((qh >> 12) & 7) + 1) as f32;
            (
                IQ1S_GRID[index],
                scale,
                scale * if qh & 0x8000 == 0 { 0.125 } else { -0.125 },
            )
        }
        21 => {
            let nib = oct * 2;
            let high = b[66 + g];
            let index1 = b[2 + nib] as usize | ((((high >> (nib % 8)) & 1) as usize) << 8);
            let index2 = b[3 + nib] as usize | ((((high >> ((nib + 1) % 8)) & 1) as usize) << 8);
            let grid = IQ3S_GRID[index1] as u64 | ((IQ3S_GRID[index2] as u64) << 32);
            let sc = (b[106 + g / 2] >> (4 * (g % 2))) & 15;
            (
                signed_grid(grid, b[74 + oct] as u32),
                packed_half::<ALIGNED>(b, 0) * (1 + 2 * sc) as f32,
                0.,
            )
        }
        22 => {
            let index = b[2 + oct] as usize | ((((b[66 + g] >> (2 * (l / 8))) & 3) as usize) << 8);
            let sc = (b[74 + g] >> (4 * (l / 16))) & 15;
            (
                signed_grid(IQ2S_GRID[index], b[34 + oct] as u32),
                packed_half::<ALIGNED>(b, 0) * (0.5 + sc as f32) * 0.25,
                0.,
            )
        }
        23 => {
            let sc = ((b[4 + g / 2] >> (4 * (g % 2))) & 15) as u16
                | (((packed_u16::<ALIGNED>(b, 2) >> (2 * g)) & 3) << 4);
            let q = (packed_word::<ALIGNED>(kind, b, 8 + g * 16 + l % 16) >> (4 * (l / 16)))
                & 0x0f0f0f0f0f0f0f0f;
            let values = iq4_values::<ALIGNED>(q);
            (
                values,
                packed_half::<ALIGNED>(b, 0) * (sc as i32 - 32) as f32,
                0.,
            )
        }
        29 => {
            let bits = (packed_u16::<ALIGNED>(b, 48) >> 12)
                | ((packed_u16::<ALIGNED>(b, 50) >> 8) & 0xf0)
                | ((packed_u16::<ALIGNED>(b, 52) >> 4) & 0xf00)
                | (packed_u16::<ALIGNED>(b, 54) & 0xf000);
            let sc =
                (packed_u16::<ALIGNED>(b, 48 + 2 * (g / 2)) >> (6 * (g % 2) + 3 * (l / 16))) & 7;
            let qh = (b[32 + oct / 2] >> (4 * (oct % 2))) & 15;
            let index = b[oct] as usize | (((qh & 7) as usize) << 8);
            let scale = half(bits) * (2 * sc + 1) as f32;
            (
                IQ1S_GRID[index],
                scale,
                scale * if qh & 8 == 0 { 0.125 } else { -0.125 },
            )
        }
        _ => unreachable!(),
    };
    (values as u32, (values >> 32) as u32, scale, offset)
}

#[cfg(test)]
mod packed_tests {
    use super::*;
    #[test]
    fn packed_coefficients_reconstruct_scalar_storage() {
        for (kind, size) in [
            (10, 84),
            (12, 144),
            (14, 210),
            (16, 66),
            (17, 74),
            (18, 98),
            (19, 50),
            (21, 110),
            (22, 82),
            (23, 136),
            (29, 56),
        ] {
            for seed in 0..17 {
                let mut b: Vec<u8> = (0..size)
                    .map(|i| ((i * 73 + seed * 37 + 19) % 256) as u8)
                    .collect();
                let offset = if kind == 10 {
                    80
                } else if kind == 14 {
                    208
                } else {
                    0
                };
                if kind == 29 {
                    for (j, nib) in [0u16, 0, 8, 3].iter().enumerate() {
                        let p = 48 + j * 2;
                        let v = u16_at(&b, p);
                        b[p..p + 2].copy_from_slice(&((v & 4095) | (nib << 12)).to_le_bytes());
                    }
                } else {
                    b[offset..offset + 2].copy_from_slice(&0x3800u16.to_le_bytes());
                    if kind == 10 || kind == 12 {
                        b[offset + 2..offset + 4].copy_from_slice(&0x3400u16.to_le_bytes());
                    }
                }
                let expected = decode_block(kind, &b).unwrap();
                let mut unaligned = vec![0u8];
                unaligned.extend_from_slice(&b);
                let decoded_unaligned = decode_block(kind, &unaligned[1..]).unwrap();
                assert_eq!(expected, decoded_unaligned);
                for i in (0..256).step_by(8) {
                    assert_eq!(
                        packed_eight(kind, &b, i),
                        packed_eight(kind, &unaligned[1..], i)
                    );
                }

                for oct in 0..32 {
                    let (lo, hi, scale, offset) = packed_eight(kind, &b, oct * 8);
                    let bytes = (lo as u64 | ((hi as u64) << 32)).to_le_bytes();
                    for j in 0..8 {
                        let actual = scale * (bytes[j] as i8) as f32 + offset;
                        let e = expected[oct * 8 + j];
                        assert!(
                            (actual - e).abs() <= 1e-6 * e.abs().max(1.),
                            "kind {kind} seed {seed} index {}: {actual} != {e}",
                            oct * 8 + j
                        );
                    }
                }
            }
        }
    }
}
