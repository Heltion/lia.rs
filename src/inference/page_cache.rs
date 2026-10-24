//! Reference-counted physical KV pages with tiered immutable prefix eviction.
use super::*;
use crate::prefix_cache::{Key, Store};
use anyhow::{Context, ensure};

#[derive(Default)]
pub(crate) struct Sequence {
    pub pages: Vec<u32>,
    keys: Vec<Key>,
}

struct Page {
    key: Option<Key>,
    users: usize,
    age: u64,
}

pub(crate) struct PageCache {
    namespace: Key,
    pages: Vec<Page>,
    prefixes: BTreeMap<Key, u32>,
    clock: u64,
    pub store: Store,
    pub gpu_hits: u64,
    pub evictions: u64,
}

impl PageCache {
    pub fn new(namespace: Key, pages: usize, store: Store) -> Result<Self> {
        ensure!(pages > 0, "empty KV page pool");
        Ok(Self {
            namespace,
            pages: (0..pages)
                .map(|_| Page {
                    key: None,
                    users: 0,
                    age: 0,
                })
                .collect(),
            prefixes: BTreeMap::new(),
            clock: 0,
            store,
            gpu_hits: 0,
            evictions: 0,
        })
    }

    pub fn attach(&mut self, engine: &mut Engine, tokens: &[u32]) -> Result<(Sequence, usize)> {
        let mut sequence = Sequence {
            pages: Vec::new(),
            keys: Vec::new(),
        };
        // Retain an uncached final token to recover next-token logits exactly.
        let count = tokens.len().saturating_sub(1) / 32;
        let mut parent = self.namespace;
        let keys: Vec<_> = tokens
            .chunks_exact(32)
            .take(count)
            .map(|tokens| {
                parent = Key::prefix(parent, tokens);
                parent
            })
            .collect();
        // Retain all resident matches before allocating missing pages. Otherwise
        // restoring an early hole could evict a later page of the same prefix.
        let mut retained: Vec<_> = keys
            .iter()
            .map(|key| {
                self.prefixes.get(key).copied().inspect(|&page| {
                    self.pages[page as usize].users += 1;
                })
            })
            .collect();
        let result = (|| -> Result<usize> {
            'prefix: for (window_index, window) in keys.chunks(16).enumerate() {
                let missing: Vec<_> = window
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| retained[window_index * 16 + index].is_none())
                    .map(|(index, &key)| (index, key))
                    .collect();
                let disk_keys: Vec<_> = missing.iter().map(|&(_, key)| key).collect();
                let loaded = self.store.load_many(&disk_keys)?;
                let mut staged: Vec<_> = (0..window.len()).map(|_| None).collect();
                for ((index, _), entry) in missing.into_iter().zip(loaded) {
                    staged[index] = Some(entry);
                }
                let mut restored = Vec::new();
                let mut stop = false;
                for (index, &key) in window.iter().enumerate() {
                    let page = if let Some(page) = retained[window_index * 16 + index].take() {
                        self.gpu_hits += 1;
                        page
                    } else {
                        let entry = staged[index].take().context("missing KV prefetch result")?;
                        if entry.data.is_none() {
                            self.store.record_access(&entry);
                            stop = true;
                            break;
                        }
                        let Some(page) = self.allocate(engine)? else {
                            stop = true;
                            break;
                        };
                        restored.push((page, key, entry));
                        page
                    };
                    self.touch(page);
                    sequence.pages.push(page);
                    sequence.keys.push(key);
                }
                if !restored.is_empty() {
                    if !stop {
                        let next = (window_index + 1) * 16;
                        let lookahead: Vec<_> = keys
                            .iter()
                            .enumerate()
                            .skip(next)
                            .take(16)
                            .filter(|(index, _)| retained[*index].is_none())
                            .map(|(_, &key)| key)
                            .collect();
                        self.store.prefetch(&lookahead)?;
                    }
                    let transfers = restored
                        .iter()
                        .map(|(page, _, entry)| {
                            Ok((
                                *page,
                                entry.data.as_deref().context("missing staged KV payload")?,
                            ))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    engine.write_kv_pages(&transfers)?;
                    for (page, key, entry) in restored {
                        self.pages[page as usize].key = Some(key);
                        self.prefixes.insert(key, page);
                        self.store.record_access(&entry);
                    }
                }
                if stop {
                    break 'prefix;
                }
            }
            Ok(sequence.pages.len() * 32)
        })();
        self.store.cancel_prefetch();
        for page in retained.into_iter().flatten() {
            self.release_page(page)?;
        }
        match result {
            Ok(cached) => Ok((sequence, cached)),
            Err(error) => {
                self.release(sequence)?;
                Err(error)
            }
        }
    }

    pub fn available_pages(&self) -> usize {
        self.pages.iter().filter(|page| page.users == 0).count()
    }

    pub fn can_fit(&self, tokens: &[u32]) -> bool {
        let mut parent = self.namespace;
        let mut shared = 0;
        for block in tokens
            .chunks_exact(32)
            .take(tokens.len().saturating_sub(1) / 32)
        {
            parent = Key::prefix(parent, block);
            if self
                .prefixes
                .get(&parent)
                .is_some_and(|&page| self.pages[page as usize].users > 0)
            {
                shared += 1;
            }
        }
        tokens.len().div_ceil(32) <= self.available_pages() + shared
    }

    /// False means active requests occupy the physical pool; the scheduler must wait or park one.
    pub fn ensure_position(
        &mut self,
        engine: &mut Engine,
        sequence: &mut Sequence,
        position: usize,
    ) -> Result<bool> {
        let additional = (position / 32 + 1).saturating_sub(sequence.pages.len());
        if additional == 0 {
            return Ok(true);
        }
        if additional > self.pages.iter().filter(|page| page.users == 0).count() {
            // Admission retries must not repeatedly allocate and release most of
            // a long request while other sequences still occupy the pool.
            return Ok(false);
        }
        while sequence.pages.len() <= position / 32 {
            let Some(page) = self.allocate(engine)? else {
                return Ok(false);
            };
            sequence.pages.push(page);
        }
        Ok(true)
    }

    pub fn seal(&mut self, sequence: &mut Sequence, tokens: &[u32], computed: usize) -> Result<()> {
        ensure!(
            computed <= tokens.len() && computed.div_ceil(32) <= sequence.pages.len(),
            "invalid computed prefix extent"
        );
        while sequence.keys.len() < computed / 32 {
            let block = sequence.keys.len();
            let key = Key::prefix(
                sequence.keys.last().copied().unwrap_or(self.namespace),
                &tokens[block * 32..(block + 1) * 32],
            );
            let current = sequence.pages[block];
            if let Some(&shared) = self.prefixes.get(&key) {
                if shared != current {
                    self.release_page(current)?;
                    self.pages[shared as usize].users += 1;
                    sequence.pages[block] = shared;
                }
            } else {
                ensure!(
                    self.pages[current as usize].key.is_none(),
                    "attempted to overwrite immutable KV page"
                );
                self.pages[current as usize].key = Some(key);
                self.prefixes.insert(key, current);
            }
            self.touch(sequence.pages[block]);
            sequence.keys.push(key);
        }
        Ok(())
    }

    pub fn release(&mut self, sequence: Sequence) -> Result<()> {
        for page in sequence.pages {
            self.release_page(page)?;
        }
        Ok(())
    }

    fn release_page(&mut self, page: u32) -> Result<()> {
        let entry = &mut self.pages[page as usize];
        ensure!(entry.users > 0, "KV page reference underflow");
        entry.users -= 1;
        Ok(())
    }

    pub fn evict_idle(&mut self, engine: &Engine) -> Result<()> {
        for page in 0..self.pages.len() {
            if self.pages[page].users == 0 && self.pages[page].key.is_some() {
                self.evict(engine, page as u32)?;
            }
        }
        Ok(())
    }

    fn allocate(&mut self, engine: &Engine) -> Result<Option<u32>> {
        let free = self
            .pages
            .iter()
            .position(|p| p.users == 0 && p.key.is_none());
        let candidate = free.or_else(|| {
            self.pages
                .iter()
                .enumerate()
                .filter(|(_, p)| p.users == 0)
                .min_by_key(|(_, p)| p.age)
                .map(|(i, _)| i)
        });
        let Some(page) = candidate else {
            return Ok(None);
        };
        if self.pages[page].key.is_some() {
            self.evict(engine, page as u32)?;
        }
        self.pages[page].users = 1;
        self.touch(page as u32);
        Ok(Some(page as u32))
    }

    fn evict(&mut self, engine: &Engine, page: u32) -> Result<()> {
        let entry = &self.pages[page as usize];
        ensure!(entry.users == 0, "cannot evict an active KV page");
        if let Some(key) = entry.key {
            let bytes = engine.read_kv_page(page)?;
            self.store.put(key, bytes.into())?;
            self.prefixes.remove(&key);
            self.pages[page as usize].key = None;
            self.evictions += 1;
        }
        Ok(())
    }

    fn touch(&mut self, page: u32) {
        self.clock += 1;
        self.pages[page as usize].age = self.clock;
    }
}
