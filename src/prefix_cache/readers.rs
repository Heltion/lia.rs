//! Persistent bounded readers amortize thread creation across prefix windows.
use super::*;
use std::sync::{Mutex, mpsc};
use std::thread::JoinHandle;

struct Job {
    index: usize,
    key: Key,
    reply: mpsc::SyncSender<(usize, Result<Arc<[u8]>>)>,
}

pub(super) struct Readers {
    jobs: Option<mpsc::SyncSender<Job>>,
    threads: Vec<JoinHandle<()>>,
}

pub(super) struct ReadBatch {
    count: usize,
    receiver: mpsc::Receiver<(usize, Result<Arc<[u8]>>)>,
}

impl ReadBatch {
    pub fn finish(self) -> Result<Vec<Result<Arc<[u8]>>>> {
        let mut payloads: Vec<_> = (0..self.count).map(|_| None).collect();
        for _ in 0..self.count {
            let (index, payload) = self.receiver.recv().context("receiving KV disk payload")?;
            let slot = payloads
                .get_mut(index)
                .context("invalid KV reader response index")?;
            ensure!(slot.is_none(), "duplicate KV reader response");
            *slot = Some(payload);
        }
        payloads
            .into_iter()
            .map(|payload| payload.context("missing KV disk payload"))
            .collect()
    }
}

impl Readers {
    pub fn new(root: std::path::PathBuf, namespace: Key, block_bytes: usize) -> Result<Self> {
        let (jobs, receiver) = mpsc::sync_channel::<Job>(16);
        let receiver = Arc::new(Mutex::new(receiver));
        let mut threads = Vec::new();
        for index in 0..16 {
            let receiver = Arc::clone(&receiver);
            let root = root.clone();
            threads.push(
                std::thread::Builder::new()
                    .name(format!("lia-kv-read-{index}"))
                    .spawn(move || {
                        loop {
                            let job = {
                                let receiver =
                                    receiver.lock().unwrap_or_else(|error| error.into_inner());
                                match receiver.recv() {
                                    Ok(job) => job,
                                    Err(_) => break,
                                }
                            };
                            let payload = std::panic::catch_unwind(|| {
                                super::disk::read_record(&root, namespace, block_bytes, job.key)
                            })
                            .unwrap_or_else(|_| Err(anyhow::anyhow!("KV disk reader panicked")));
                            let _ = job.reply.send((job.index, payload));
                        }
                    })
                    .context("starting KV disk reader")?,
            );
        }
        Ok(Self {
            jobs: Some(jobs),
            threads,
        })
    }

    pub fn read_many(&self, keys: &[Key]) -> Result<Vec<Result<Arc<[u8]>>>> {
        self.start(keys)?.finish()
    }

    pub fn start(&self, keys: &[Key]) -> Result<ReadBatch> {
        ensure!(keys.len() <= 16, "KV reader window exceeds staging bound");
        let (reply, receiver) = mpsc::sync_channel(keys.len().max(1));
        let jobs = self.jobs.as_ref().context("KV reader pool is closed")?;
        for (index, &key) in keys.iter().enumerate() {
            jobs.send(Job {
                index,
                key,
                reply: reply.clone(),
            })
            .map_err(|_| anyhow::anyhow!("KV reader pool stopped"))?;
        }
        drop(reply);
        Ok(ReadBatch {
            count: keys.len(),
            receiver,
        })
    }
}

impl Drop for Readers {
    fn drop(&mut self) {
        self.jobs.take();
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}
