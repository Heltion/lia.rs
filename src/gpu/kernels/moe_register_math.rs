//! Register operands for signed W2 coefficients and two-piece BF16 inputs.
use cuda_device::convert;

#[inline(always)]
pub(super) fn coefficients(codes: u32, scale: u32) -> u32 {
    // Nonzero normal BF16 scales, with exponent <= 253, represent all
    // four signed coefficients exactly. Callers must validate this range.
    let negative = ((scale + 128) ^ 0x8000) | ((scale ^ 0x8000) << 16);
    let positive = (scale & 0x8000) | (scale << 16);
    cuda_device::prmt::prmt(
        negative,
        positive,
        0x1010 + (codes & 3) * 0x22 + ((codes >> 2) & 3) * 0x2200,
    )
}

#[inline(always)]
pub(super) fn split(a: f32, b: f32) -> (u32, u32) {
    let high = convert::cvt_bf16x2_f32(a, b);
    let (ah, bh) = convert::cvt_f32x2_bf16x2(high);
    (high, convert::cvt_bf16x2_f32(a - ah, b - bh))
}

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
    // SAFETY: upload gives an aligned u16 allocation, represented as bytes;
    // complete tiles and bounded expert indices cover this word.
    let word = unsafe { weights.as_ptr().cast::<u16>().add(i).read() } as u32;
    let at = (tile * (columns / 128) + k / 128) * 16 + lane / 4;
    let s = scales[at] as u32;
    let t = scales[at + 8] as u32;
    decode(word, s, t)
}

#[inline(always)]
pub(super) fn decode(word: u32, s: u32, t: u32) -> [u32; 4] {
    [
        coefficients(word, s),
        coefficients(word >> 4, t),
        coefficients(word >> 8, s),
        coefficients(word >> 12, t),
    ]
}
