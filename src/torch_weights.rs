//! Read the stored FloatStorage state dictionaries used by MiniCPM-o's codec.
//! This interprets a bounded data-only subset, never imports model code.
use anyhow::{Context, Result, ensure};
use std::{collections::BTreeMap, path::Path};
mod pickle;
mod value;
mod zip;
use value::{Storage, Value};

#[derive(Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    offset: usize,
    elements: usize,
    storage: Storage,
}
impl Tensor {
    fn new(shape: Vec<usize>, stride: Vec<usize>, offset: usize, storage: Storage) -> Result<Self> {
        ensure!(
            shape.len() == stride.len() && shape.len() <= 8,
            "unsupported tensor rank"
        );
        let mut elements = 1usize;
        for (&dim, &step) in shape.iter().zip(&stride).rev() {
            ensure!(
                dim <= 1 || step == elements,
                "noncontiguous FloatStorage tensor"
            );
            elements = elements
                .checked_mul(dim)
                .context("tensor element count overflow")?;
        }
        ensure!(
            offset <= storage.elements && elements <= storage.elements - offset,
            "tensor exceeds FloatStorage extent"
        );
        Ok(Self {
            shape,
            stride,
            offset,
            elements,
            storage,
        })
    }
}
pub struct Archive {
    zip: zip::StoredZip,
    prefix: String,
    tensors: BTreeMap<String, Tensor>,
}
impl Archive {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let zip = zip::StoredZip::open(path.as_ref())?;
        let manifests: Vec<_> = zip
            .entries
            .keys()
            .filter(|s| s.ends_with("/data.pkl"))
            .collect();
        ensure!(
            manifests.len() == 1,
            "expected one PyTorch state dictionary"
        );
        let manifest = manifests[0].clone();
        let prefix = manifest.strip_suffix("data.pkl").unwrap().to_owned();
        ensure!(
            zip.read(&format!("{prefix}byteorder"), 16)? == b"little",
            "unsupported tensor byte order"
        );
        let bytes = zip.read(&manifest, 16 * 1024 * 1024)?;
        let root = pickle::parse(&bytes)?.items()?;
        let mut tensors = BTreeMap::new();
        for (name, value) in root {
            let tensor = match &*value.0.borrow() {
                Value::Tensor(t) => t.clone(),
                _ => anyhow::bail!("non-tensor state entry {name}"),
            };
            let entry = zip
                .entries
                .get(&format!("{prefix}data/{}", tensor.storage.key))
                .with_context(|| format!("missing storage for {name}"))?;
            ensure!(
                tensor.storage.elements.checked_mul(4) == Some(entry.size),
                "FloatStorage size mismatch for {name}"
            );
            tensors.insert(name, tensor);
        }
        ensure!(!tensors.is_empty(), "empty tensor state dictionary");
        Ok(Self {
            zip,
            prefix,
            tensors,
        })
    }
    pub fn tensors(&self) -> &BTreeMap<String, Tensor> {
        &self.tensors
    }
    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .with_context(|| format!("missing Torch weight {name}"))
    }
    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        let tensor = self.tensor(name)?;
        let data = self.zip.read(
            &format!("{}data/{}", self.prefix, tensor.storage.key),
            2 * 1024 * 1024 * 1024,
        )?;
        let start = tensor.offset * 4;
        let count = tensor.elements * 4;
        if start == 0 && count == data.len() {
            Ok(data)
        } else {
            Ok(data[start..start + count].to_vec())
        }
    }
}
