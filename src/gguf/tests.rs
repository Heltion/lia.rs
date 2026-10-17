use super::*;
use std::sync::atomic::{AtomicU64, Ordering};
static NEXT: AtomicU64 = AtomicU64::new(0);
fn open_bytes(bytes: &[u8]) -> anyhow::Result<Model> {
    let path = std::env::temp_dir().join(format!(
        "lia-gguf-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, bytes)?;
    let result = Model::open(&path);
    std::fs::remove_file(path)?;
    result
}
fn tiny() -> Vec<u8> {
    let mut b = b"GGUF".to_vec();
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes()); // tensor count
    b.extend_from_slice(&0u64.to_le_bytes()); // metadata count
    b.extend_from_slice(&1u64.to_le_bytes());
    b.push(b'x');
    b.extend_from_slice(&1u32.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes()); // one f32
    b.extend_from_slice(&0u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes()); // data offset
    b.resize(64, 0);
    b.extend_from_slice(&1.25f32.to_le_bytes());
    b
}
#[test]
fn valid_container_and_truncation() {
    let b = tiny();
    let model = open_bytes(&b).unwrap();
    assert_eq!(model.data_offset, 64);
    assert_eq!(model.tensors[0].byte_len, 4);
    for end in 0..b.len() {
        assert!(
            open_bytes(&b[..end]).is_err(),
            "accepted truncation at {end}"
        );
    }
}
#[test]
fn rejects_invalid_header_dimension_type_and_extent() {
    for (offset, replacement) in [(0, b'X'), (4, 1), (33, 0), (45, 255), (49, 32)] {
        let mut b = tiny();
        b[offset] = replacement;
        assert!(open_bytes(&b).is_err(), "accepted invalid byte at {offset}");
    }
}
