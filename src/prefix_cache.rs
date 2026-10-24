//! Bounded, model-scoped KV block storage shared by serial and batched scheduling.
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path, sync::Arc};
mod disk;
mod readers;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Key(pub [u8; 32]);

impl Key {
    pub fn variant(self, profile: &[u8]) -> Self {
        let mut hash = Sha256::new();
        hash.update(b"lia-kv-numerics-v1");
        hash.update(self.0);
        hash.update((profile.len() as u64).to_le_bytes());
        hash.update(profile);
        Self(hash.finalize().into())
    }
    pub fn checkpoint(directory: &Path) -> Result<Self> {
        use anyhow::Context;
        use std::io::Read;
        let index: serde_json::Value = serde_json::from_reader(std::fs::File::open(
            directory.join("model.safetensors.index.json"),
        )?)?;
        let mut files = std::collections::BTreeSet::new();
        for value in index["weight_map"]
            .as_object()
            .context("missing checkpoint weight map")?
            .values()
        {
            let file = value.as_str().context("invalid shard name")?;
            ensure!(
                Path::new(file).file_name().and_then(|n| n.to_str()) == Some(file),
                "invalid shard path"
            );
            files.insert(file.to_owned());
        }
        files.extend(["config.json".to_owned(), "tokenizer.json".to_owned()]);
        let mut hash = Sha256::new();
        hash.update(b"lia-minicpm5-f16-kv32-v1");
        let mut buffer = vec![0_u8; 128 * 1024];
        for file in files {
            hash.update((file.len() as u64).to_le_bytes());
            hash.update(file.as_bytes());
            let mut input = std::fs::File::open(directory.join(&file))
                .with_context(|| format!("opening cache identity input {file}"))?;
            hash.update(input.metadata()?.len().to_le_bytes());
            loop {
                let read = input
                    .read(&mut buffer)
                    .with_context(|| format!("hashing checkpoint {file}"))?;
                if read == 0 {
                    break;
                }
                hash.update(&buffer[..read]);
            }
        }
        Ok(Self(hash.finalize().into()))
    }
    pub fn prefix(parent: Self, tokens: &[u32]) -> Self {
        let mut hash = Sha256::new();
        hash.update(b"lia-kv-prefix-v1");
        hash.update(parent.0);
        hash.update((tokens.len() as u64).to_le_bytes());
        for token in tokens {
            hash.update(token.to_le_bytes());
        }
        Self(hash.finalize().into())
    }
    fn hex(self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }
}

#[derive(Default, Debug)]
pub(crate) struct Statistics {
    pub ram_hits: u64,
    pub disk_hits: u64,
    pub misses: u64,
}

pub(crate) struct Loaded {
    pub data: Option<Arc<[u8]>>,
    disk: bool,
}

pub(crate) struct Store {
    ram: BTreeMap<Key, (Arc<[u8]>, u64)>,
    ram_limit: usize,
    ram_bytes: usize,
    clock: u64,
    disk: disk::Disk,
    pub statistics: Statistics,
}

impl Store {
    pub fn open(
        root: &Path,
        namespace: Key,
        ram_limit: usize,
        disk_limit: u64,
        block_bytes: usize,
    ) -> Result<Self> {
        ensure!(block_bytes > 0, "KV block size must be positive");
        ensure!(
            ram_limit >= block_bytes || disk_limit >= block_bytes as u64 + 112,
            "KV cache budgets cannot hold one block"
        );
        Ok(Self {
            ram: BTreeMap::new(),
            ram_limit,
            ram_bytes: 0,
            clock: 0,
            disk: disk::Disk::open(root, namespace, disk_limit, block_bytes)?,
            statistics: Statistics::default(),
        })
    }

    #[cfg(test)]
    pub fn get(&mut self, key: Key) -> Result<Option<Arc<[u8]>>> {
        let entry = self.load_many(&[key])?.pop().expect("one KV lookup");
        self.record_access(&entry);
        Ok(entry.data)
    }

    /// One lookahead window overlaps disk validation with the current GPU upload.
    pub fn prefetch(&mut self, keys: &[Key]) -> Result<()> {
        ensure!(keys.len() <= 16, "KV prefetch window exceeds staging bound");
        let missing: Vec<_> = keys
            .iter()
            .copied()
            .filter(|key| !self.ram.contains_key(key))
            .collect();
        self.disk.prefetch(&missing)
    }

    pub fn cancel_prefetch(&mut self) {
        self.disk.cancel_prefetch();
    }

    /// Each read and lookahead window holds at most sixteen payloads.
    /// Counters are committed only when the caller actually consumes an entry.
    pub fn load_many(&mut self, keys: &[Key]) -> Result<Vec<Loaded>> {
        ensure!(keys.len() <= 16, "KV read window exceeds staging bound");
        let mut entries = Vec::with_capacity(keys.len());
        let mut missing = Vec::new();
        for (index, &key) in keys.iter().enumerate() {
            self.clock += 1;
            let data = self.ram.get_mut(&key).map(|(data, age)| {
                *age = self.clock;
                Arc::clone(data)
            });
            if data.is_none() {
                missing.push((index, key));
            }
            entries.push(Loaded { data, disk: false });
        }
        let disk_keys: Vec<_> = missing.iter().map(|&(_, key)| key).collect();
        let values = self.disk.read_many(&disk_keys)?;
        for ((index, key), data) in missing.into_iter().zip(values) {
            if let Some(data) = &data {
                self.remember(key, Arc::clone(data))?;
            }
            entries[index] = Loaded { data, disk: true };
        }
        Ok(entries)
    }

    pub fn record_access(&mut self, entry: &Loaded) {
        if entry.data.is_none() {
            self.statistics.misses += 1;
        } else if entry.disk {
            self.statistics.disk_hits += 1;
        } else {
            self.statistics.ram_hits += 1;
        }
    }

    pub fn put(&mut self, key: Key, data: Arc<[u8]>) -> Result<()> {
        if data.len() > self.ram_limit {
            return self.disk.write(key, &data);
        }
        self.clock += 1;
        self.remember(key, data)
    }

    fn remember(&mut self, key: Key, data: Arc<[u8]>) -> Result<()> {
        if data.len() > self.ram_limit {
            return Ok(());
        }
        if let Some((previous, _)) = self.ram.remove(&key) {
            self.ram_bytes -= previous.len();
        }
        while self.ram_bytes + data.len() > self.ram_limit {
            let oldest = self
                .ram
                .iter()
                .min_by_key(|(_, (_, age))| *age)
                .map(|(key, _)| *key);
            let Some(oldest) = oldest else {
                break;
            };
            let (payload, _) = self.ram.get(&oldest).expect("selected RAM victim");
            self.disk
                .write(oldest, payload)
                .context("spilling RAM KV cache to SSD")?;
            if let Some((removed, _)) = self.ram.remove(&oldest) {
                self.ram_bytes -= removed.len();
            }
        }
        self.ram_bytes += data.len();
        self.ram.insert(key, (data, self.clock));
        Ok(())
    }

    pub fn clear_ram(&mut self) -> Result<()> {
        let mut order: Vec<_> = self
            .ram
            .iter()
            .map(|(&key, (_, age))| (*age, key))
            .collect();
        order.sort_unstable();
        for (_, key) in order {
            self.disk
                .write(key, &self.ram[&key].0)
                .context("flushing RAM KV cache to SSD")?;
        }
        self.ram.clear();
        self.ram_bytes = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    struct Directory(std::path::PathBuf);
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn directory() -> Directory {
        Directory(std::env::temp_dir().join(format!(
            "lia-kv-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )))
    }

    #[test]
    fn ram_eviction_disk_restart_and_namespaces() -> Result<()> {
        let root = directory();
        let namespace = Key([9; 32]);
        let a = Key::prefix(namespace, &[1, 2]);
        let b = Key::prefix(namespace, &[1, 3]);
        assert_ne!(a, b);
        let mut store = Store::open(&root.0, namespace, 4, 1024, 4)?;
        store.put(a, Arc::from([1_u8, 2, 3, 4]))?;
        assert_eq!(&*store.get(a)?.expect("RAM hit"), &[1, 2, 3, 4]);
        store.put(b, Arc::from([5_u8, 6, 7, 8]))?;
        assert_eq!(&*store.get(a)?.expect("disk restore"), &[1, 2, 3, 4]);
        assert_eq!(store.statistics.ram_hits, 1);
        assert_eq!(store.statistics.disk_hits, 1);
        assert!(Store::open(&root.0, namespace, 4, 1024, 4).is_err());
        drop(store);
        let mut restored = Store::open(&root.0, namespace, 0, 1024, 4)?;
        assert_eq!(&*restored.get(b)?.expect("restart restore"), &[5, 6, 7, 8]);
        let mut other = Store::open(&root.0, Key([8; 32]), 4, 1024, 4)?;
        assert!(other.get(a)?.is_none());
        Ok(())
    }

    #[test]
    fn corrupted_entries_are_misses_and_disk_is_bounded() -> Result<()> {
        let root = directory();
        let namespace = Key([7; 32]);
        let a = Key::prefix(namespace, &[1]);
        let b = Key::prefix(namespace, &[2]);
        let mut store = Store::open(&root.0, namespace, 0, 116, 4)?;
        store.put(a, Arc::from([1_u8; 4]))?;
        store.put(b, Arc::from([2_u8; 4]))?;
        assert!(store.get(a)?.is_none());
        let file = root
            .0
            .join(namespace.hex())
            .join(format!("{}.lia-kv", b.hex()));
        std::fs::write(file, b"corrupt")?;
        assert!(store.get(b)?.is_none());
        Ok(())
    }
    #[test]
    fn writeback_flush_parallel_reads_and_payload_corruption() -> Result<()> {
        use std::io::{Seek, SeekFrom, Write};
        let root = directory();
        let namespace = Key([6; 32]);
        let keys: Vec<_> = (0..16).map(|i| Key::prefix(namespace, &[i])).collect();
        let mut store = Store::open(&root.0, namespace, 16 * 4096, 128 * 1024, 4096)?;
        for (index, &key) in keys.iter().enumerate() {
            store.put(key, vec![index as u8; 4096].into())?;
        }
        let namespace_path = root.0.join(namespace.hex());
        assert_eq!(
            std::fs::read_dir(&namespace_path)?.count(),
            1,
            "RAM entries wrote SSD early"
        );
        store.clear_ram()?;
        assert_eq!(store.ram_bytes, 0);
        let corrupt = namespace_path.join(format!("{}.lia-kv", keys[3].hex()));
        let mut file = std::fs::OpenOptions::new().write(true).open(&corrupt)?;
        file.seek(SeekFrom::Start(112 + 257))?;
        file.write_all(&[0xff])?;
        drop(file);
        store.prefetch(&keys)?;
        assert_eq!(store.statistics.disk_hits, 0);
        let entries = store.load_many(&keys)?;
        assert_eq!(
            store.statistics.disk_hits, 0,
            "prefetch counted unconsumed hits"
        );
        for (index, entry) in entries.iter().enumerate() {
            if index == 3 {
                assert!(entry.data.is_none());
            } else {
                assert_eq!(
                    &**entry.data.as_ref().expect("validated payload"),
                    vec![index as u8; 4096]
                );
            }
            store.record_access(entry);
        }
        assert_eq!(store.statistics.disk_hits, 15);
        assert_eq!(store.statistics.misses, 1);
        assert!(!corrupt.exists(), "invalid payload stayed indexed");
        assert!(store.load_many(&vec![keys[0]; 17]).is_err());
        assert!(store.prefetch(&vec![keys[0]; 17]).is_err());
        store.clear_ram()?;
        // An abandoned lookahead must not constrain a later divergent lookup.
        store.prefetch(&keys)?;
        let divergent = store.load_many(&[keys[15], keys[0]])?;
        assert_eq!(&**divergent[0].data.as_ref().unwrap(), vec![15; 4096]);
        assert_eq!(&**divergent[1].data.as_ref().unwrap(), vec![0; 4096]);
        store.prefetch(&keys)?;
        store.cancel_prefetch();
        let repeated = store.load_many(&keys)?;
        assert_eq!(
            repeated.iter().filter(|entry| entry.data.is_some()).count(),
            15,
            "persistent readers changed reused-window results"
        );
        Ok(())
    }
}
