//! Checksummed atomic SSD records; a directory lock protects the capacity ledger.
use super::*;
use anyhow::Context;
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::PathBuf,
};
const MAGIC: &[u8; 8] = b"LIAKV001";
const HEADER: usize = 112;

pub(super) struct Disk {
    root: PathBuf,
    namespace: Key,
    limit: u64,
    bytes: u64,
    block_bytes: usize,
    clock: u64,
    records: BTreeMap<Key, (u64, u64)>,
    pending: Option<(Vec<Key>, super::readers::ReadBatch)>,
    readers: super::readers::Readers,
    _lock: File,
}

impl Disk {
    pub fn open(root: &Path, namespace: Key, limit: u64, block_bytes: usize) -> Result<Self> {
        let root = root.join(namespace.hex());
        fs::create_dir_all(&root).context("creating KV disk cache directory")?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(root.join("lock"))
            .context("opening KV cache lock")?;
        lock.try_lock()
            .context("locking KV cache directory (already in use)")?;
        let readers = super::readers::Readers::new(root.clone(), namespace, block_bytes)?;
        let mut cache = Self {
            root,
            namespace,
            limit,
            bytes: 0,
            block_bytes,
            clock: 0,
            records: BTreeMap::new(),
            pending: None,
            _lock: lock,
            readers,
        };
        for entry in fs::read_dir(&cache.root).context("listing KV disk cache")? {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            // A previous interrupted write is never a published cache record.
            // The exclusive lock guarantees no live writer owns these files.
            if name.starts_with("pending-") {
                fs::remove_file(entry.path()).context("removing interrupted KV write")?;
                continue;
            }
            let Some(hex) = name.strip_suffix(".lia-kv") else {
                continue;
            };
            let Some(key) = parse_key(hex) else {
                continue;
            };
            let size = entry.metadata()?.len();
            cache.clock += 1;
            cache.records.insert(key, (size, cache.clock));
            cache.bytes += size;
        }
        cache.trim(0)?;
        Ok(cache)
    }

    fn path(&self, key: Key) -> PathBuf {
        self.root.join(format!("{}.lia-kv", key.hex()))
    }

    pub fn read_many(&mut self, keys: &[Key]) -> Result<Vec<Option<Arc<[u8]>>>> {
        ensure!(keys.len() <= 16, "KV read window exceeds staging bound");
        let present: Vec<_> = keys
            .iter()
            .enumerate()
            .filter(|(_, key)| self.records.contains_key(*key))
            .map(|(index, &key)| (index, key))
            .collect();
        let disk_keys: Vec<_> = present.iter().map(|&(_, key)| key).collect();
        let values = match self.pending.take() {
            Some((keys, batch)) if keys == disk_keys => batch.finish()?,
            _ => self.readers.read_many(&disk_keys)?,
        };
        let mut decoded: Vec<_> = (0..keys.len()).map(|_| Ok(None)).collect();
        for ((index, _), value) in present.into_iter().zip(values) {
            decoded[index] = value.map(Some);
        }
        let mut result = Vec::with_capacity(keys.len());
        for (&key, value) in keys.iter().zip(decoded) {
            match value {
                Ok(Some(data)) => {
                    self.clock += 1;
                    self.records.get_mut(&key).expect("indexed KV record").1 = self.clock;
                    result.push(Some(data));
                }
                Ok(None) => result.push(None),
                Err(_) => {
                    self.remove(key)?;
                    result.push(None);
                }
            }
        }
        Ok(result)
    }

    pub fn prefetch(&mut self, keys: &[Key]) -> Result<()> {
        ensure!(keys.len() <= 16, "KV prefetch window exceeds staging bound");
        self.pending.take();
        let keys: Vec<_> = keys
            .iter()
            .copied()
            .filter(|key| self.records.contains_key(key))
            .collect();
        if !keys.is_empty() {
            let batch = self.readers.start(&keys)?;
            self.pending = Some((keys, batch));
        }
        Ok(())
    }

    pub fn cancel_prefetch(&mut self) {
        self.pending.take();
    }

    pub fn write(&mut self, key: Key, data: &[u8]) -> Result<()> {
        ensure!(
            data.len() == self.block_bytes,
            "invalid KV block payload size"
        );
        let size = (HEADER + data.len()) as u64;
        if size > self.limit {
            return Ok(());
        }
        if let Some(record) = self.records.get_mut(&key) {
            // Prefix records are immutable. Keep the first persisted value.
            self.clock += 1;
            record.1 = self.clock;
            return Ok(());
        }
        self.trim(size)?;
        self.clock += 1;
        let temporary = self
            .root
            .join(format!("pending-{}-{}", std::process::id(), self.clock));
        let result = (|| -> Result<()> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)?;
            file.write_all(MAGIC)?;
            file.write_all(&self.namespace.0)?;
            file.write_all(&key.0)?;
            file.write_all(&(data.len() as u64).to_le_bytes())?;
            file.write_all(&Sha256::digest(data))?;
            file.write_all(data)?;
            file.sync_data().context("persisting KV record")?;
            fs::rename(&temporary, self.path(key)).context("publishing KV record")?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(temporary);
        }
        result?;
        self.records.insert(key, (size, self.clock));
        self.bytes += size;
        Ok(())
    }

    fn remove(&mut self, key: Key) -> Result<()> {
        match fs::remove_file(self.path(key)) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e).context("removing KV cache record"),
        }
        if let Some((size, _)) = self.records.remove(&key) {
            self.bytes -= size;
        }
        Ok(())
    }

    fn trim(&mut self, required: u64) -> Result<()> {
        while self.bytes.saturating_add(required) > self.limit {
            let Some(key) = self
                .records
                .iter()
                .min_by_key(|(_, (_, age))| *age)
                .map(|(k, _)| *k)
            else {
                break;
            };
            self.remove(key)?;
        }
        Ok(())
    }
}

pub(super) fn read_record(
    root: &Path,
    namespace: Key,
    block_bytes: usize,
    key: Key,
) -> Result<Arc<[u8]>> {
    let mut file = File::open(root.join(format!("{}.lia-kv", key.hex())))
        .context("opening KV cache record")?;
    ensure!(
        file.metadata()?.len() == (HEADER + block_bytes) as u64,
        "invalid KV record extent"
    );
    let mut header = [0_u8; HEADER];
    file.read_exact(&mut header)?;
    ensure!(
        &header[..8] == MAGIC && header[8..40] == namespace.0 && header[40..72] == key.0,
        "incompatible KV record identity"
    );
    ensure!(
        u64::from_le_bytes(header[72..80].try_into()?) == block_bytes as u64,
        "KV layout mismatch"
    );
    // Allocate the shared payload directly, avoiding a full Vec-to-Arc copy.
    let mut uninitialized = Arc::<[u8]>::new_uninit_slice(block_bytes);
    let bytes = Arc::get_mut(&mut uninitialized).context("initializing KV read buffer")?;
    // SAFETY: the uniquely owned allocation covers block_bytes bytes; zero is
    // valid for every u8. Read::read_exact receives initialized bytes below.
    let mut data = unsafe {
        bytes.as_mut_ptr().write_bytes(0, block_bytes);
        uninitialized.assume_init()
    };
    file.read_exact(Arc::get_mut(&mut data).context("filling KV read buffer")?)?;
    ensure!(
        Sha256::digest(&data)[..] == header[80..112],
        "KV checksum mismatch"
    );
    Ok(data)
}

fn parse_key(hex: &str) -> Option<Key> {
    if hex.len() != 64 || !hex.is_ascii() {
        return None;
    }
    let mut bytes = [0_u8; 32];
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(Key(bytes))
}
