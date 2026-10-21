use super::{Shard, Tensor};
use anyhow::{Context, Result, ensure};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

pub struct Archive {
    root: PathBuf,
    index: BTreeMap<String, String>,
    shards: BTreeMap<String, Shard>,
}

impl Archive {
    pub fn open(directory: impl AsRef<Path>) -> Result<Self> {
        let root = directory.as_ref().to_path_buf();
        let path = root.join("model.safetensors.index.json");
        let value: serde_json::Value = serde_json::from_reader(
            File::open(&path).with_context(|| format!("opening {}", path.display()))?,
        )
        .context("reading safetensors shard index")?;
        let map = value["weight_map"]
            .as_object()
            .context("missing safetensors weight_map")?;
        ensure!(!map.is_empty(), "empty safetensors shard index");
        let mut index = BTreeMap::new();
        let mut shards = BTreeMap::new();
        for (name, file) in map {
            let file = file.as_str().context("non-string safetensors shard name")?;
            ensure!(
                Path::new(file).file_name().and_then(|p| p.to_str()) == Some(file)
                    && file.ends_with(".safetensors"),
                "invalid safetensors shard filename"
            );
            if !shards.contains_key(file) {
                shards.insert(file.to_owned(), Shard::open(root.join(file))?);
            }
            ensure!(
                shards[file].tensors.contains_key(name),
                "indexed tensor {name} missing from {file}"
            );
            index.insert(name.clone(), file.to_owned());
        }
        for (file, shard) in &shards {
            for name in shard.tensors.keys() {
                ensure!(
                    index.get(name) == Some(file),
                    "tensor {name} has no matching shard index entry"
                );
            }
        }
        Ok(Self {
            root,
            index,
            shards,
        })
    }

    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        let shard = self
            .index
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        Ok(&self.shards[shard].tensors[name])
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(String::as_str)
    }

    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        let tensor = self.tensor(name)?;
        let shard = &self.index[name];
        let mut file = File::open(self.root.join(shard))?;
        let offset = self.shards[shard]
            .data_offset
            .checked_add(tensor.offset)
            .context("safetensors absolute offset overflow")?;
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes = vec![0; tensor.byte_len];
        file.read_exact(&mut bytes)
            .with_context(|| format!("reading {name} from {shard}"))?;
        Ok(bytes)
    }
}
