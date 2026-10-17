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
