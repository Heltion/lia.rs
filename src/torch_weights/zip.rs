//! Bounded uncompressed ZIP records; no files are extracted from the archive.
use anyhow::{Context, Result, ensure};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
    sync::Mutex,
};

pub(super) struct Entry {
    pub size: usize,
    offset: u64,
    crc: u32,
}
pub(super) struct StoredZip {
    file: Mutex<File>,
    pub entries: BTreeMap<String, Entry>,
}
fn word(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap())
}
fn dword(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}
fn read_at(file: &mut File, offset: u64, length: usize) -> Result<Vec<u8>> {
    file.seek(SeekFrom::Start(offset))?;
    let mut bytes = vec![0; length];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}
impl StoredZip {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let length = file.metadata()?.len();
        ensure!(length >= 22, "truncated ZIP archive");
        let tail_len = length.min(65557) as usize;
        let tail = read_at(&mut file, length - tail_len as u64, tail_len)?;
        let end = (0..=tail_len - 22)
            .rev()
            .find(|&i| {
                tail[i..i + 4] == *b"PK\x05\x06"
                    && i + 22 + usize::from(word(&tail, i + 20)) == tail_len
            })
            .context("missing ZIP directory end")?;
        ensure!(
            word(&tail, end + 4) == 0
                && word(&tail, end + 6) == 0
                && word(&tail, end + 8) == word(&tail, end + 10),
            "multidisk ZIP is unsupported"
        );
        let count = usize::from(word(&tail, end + 10));
        let directory_size = dword(&tail, end + 12) as usize;
        let directory_offset = u64::from(dword(&tail, end + 16));
        ensure!(
            count > 0
                && count < 65535
                && directory_size <= 32 * 1024 * 1024
                && directory_offset
                    .checked_add(directory_size as u64)
                    .is_some_and(|n| n <= length - tail_len as u64 + end as u64),
            "invalid or unsupported ZIP64 directory"
        );
        let directory = read_at(&mut file, directory_offset, directory_size)?;
        let mut entries = BTreeMap::new();
        let mut ranges = Vec::new();
        let mut at = 0;
        for _ in 0..count {
            ensure!(
                directory.len() - at >= 46 && directory[at..at + 4] == *b"PK\x01\x02",
                "invalid ZIP directory record"
            );
            let header = &directory[at..at + 46];
            let flags = word(header, 8);
            let size = dword(header, 24) as usize;
            ensure!(
                flags & 1 == 0
                    && word(header, 10) == 0
                    && dword(header, 20) as usize == size
                    && size < u32::MAX as usize
                    && word(header, 34) == 0,
                "expected unencrypted stored ZIP entry"
            );
            let name_len = usize::from(word(header, 28));
            let total =
                46 + name_len + usize::from(word(header, 30)) + usize::from(word(header, 32));
            ensure!(
                total <= directory.len() - at && name_len > 0,
                "truncated ZIP entry name"
            );
            let name_bytes = &directory[at + 46..at + 46 + name_len];
            let name = std::str::from_utf8(name_bytes)
                .context("invalid ZIP entry name")?
                .to_owned();
            let local = u64::from(dword(header, 42));
            ensure!(
                local.checked_add(30).is_some_and(|n| n <= directory_offset),
                "invalid ZIP local offset"
            );
            let local_header = read_at(&mut file, local, 30)?;
            ensure!(
                local_header[..4] == *b"PK\x03\x04"
                    && word(&local_header, 6) == flags
                    && word(&local_header, 8) == 0
                    && usize::from(word(&local_header, 26)) == name_len,
                "ZIP local header mismatch"
            );
            ensure!(
                read_at(&mut file, local + 30, name_len)? == name_bytes,
                "ZIP local name mismatch"
            );
            let offset = local + 30 + name_len as u64 + u64::from(word(&local_header, 28));
            let stop = offset
                .checked_add(size as u64)
                .context("ZIP extent overflow")?;
            ensure!(stop <= directory_offset, "ZIP data exceeds storage area");
            ranges.push((local, stop));
            ensure!(
                entries
                    .insert(
                        name,
                        Entry {
                            size,
                            offset,
                            crc: dword(header, 16)
                        }
                    )
                    .is_none(),
                "duplicate ZIP entry"
            );
            at += total;
        }
        ensure!(at == directory.len(), "unexpected ZIP directory suffix");
        ranges.sort_unstable();
        ensure!(
            ranges.windows(2).all(|w| w[0].1 <= w[1].0),
            "overlapping ZIP records"
        );
        Ok(Self {
            file: Mutex::new(file),
            entries,
        })
    }
    pub fn read(&self, name: &str, maximum: usize) -> Result<Vec<u8>> {
        let entry = self
            .entries
            .get(name)
            .with_context(|| format!("missing ZIP entry {name}"))?;
        ensure!(entry.size <= maximum, "ZIP entry {name} exceeds read limit");
        let mut file = self
            .file
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned weight archive file"))?;
        let data = read_at(&mut file, entry.offset, entry.size)
            .with_context(|| format!("reading ZIP entry {name}"))?;
        ensure!(
            crc32(&data) == entry.crc,
            "ZIP checksum mismatch for {name}"
        );
        Ok(data)
    }
}
fn crc32(bytes: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, entry) in table.iter_mut().enumerate() {
        let mut crc = i as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ if crc & 1 != 0 { 0xedb88320 } else { 0 };
        }
        *entry = crc;
    }
    let mut crc = u32::MAX;
    for &byte in bytes {
        crc = table[((crc as u8) ^ byte) as usize] ^ (crc >> 8);
    }
    !crc
}
#[cfg(test)]
mod tests {
    #[test]
    fn crc_matches_standard_check_vector() {
        assert_eq!(super::crc32(b"123456789"), 0xcbf43926);
    }
}
