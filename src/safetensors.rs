//! Checked, little-endian safetensors loading without executing model code.
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::{collections::BTreeMap, fs::File, io::Read, path::Path};

mod archive;
pub use archive::Archive;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    Bf16,
    F16,
    F32,
    I32,
    I64,
    I8,
    U8,
}

impl Dtype {
    pub fn bytes(self) -> usize {
        match self {
            Self::Bf16 | Self::F16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::I64 => 8,
            Self::I8 | Self::U8 => 1,
        }
    }

    fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "BF16" => Self::Bf16,
            "F16" => Self::F16,
            "F32" => Self::F32,
            "I32" => Self::I32,
            "I64" => Self::I64,
            "I8" => Self::I8,
            "U8" => Self::U8,
            _ => anyhow::bail!("unsupported safetensors dtype {name}"),
        })
    }
}

#[derive(Debug)]
pub struct Tensor {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub byte_len: usize,
}

pub struct Shard {
    pub tensors: BTreeMap<String, Tensor>,
    pub data_offset: u64,
}

fn integer(value: &Value) -> Result<usize> {
    usize::try_from(
        value
            .as_u64()
            .context("expected unsigned tensor dimension or offset")?,
    )
    .context("tensor extent exceeds address space")
}

impl Shard {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let size = file.metadata()?.len();
        let mut bytes = [0; 8];
        file.read_exact(&mut bytes)
            .context("reading safetensors header size")?;
        let length = u64::from_le_bytes(bytes);
        ensure!(
            length > 0 && length <= 100_000_000 && length <= size.saturating_sub(8),
            "invalid safetensors header length"
        );
        let mut header = vec![0; length as usize];
        file.read_exact(&mut header)
            .context("reading safetensors header")?;
        Self::parse(&header, size - 8 - length)
            .with_context(|| format!("validating {}", path.display()))
    }

    fn parse(header: &[u8], data_len: u64) -> Result<Self> {
        ensure!(
            header.first() == Some(&b'{'),
            "safetensors header must start with an object"
        );
        let json: Value =
            serde_json::from_slice(header).context("parsing safetensors header JSON")?;
        let object = json
            .as_object()
            .context("expected safetensors header object")?;
        let mut tensors = BTreeMap::new();
        let mut spans = Vec::new();
        for (name, value) in object {
            if name == "__metadata__" {
                let metadata = value.as_object().context("invalid safetensors metadata")?;
                ensure!(
                    metadata.values().all(Value::is_string),
                    "non-string safetensors metadata"
                );
                continue;
            }
            let dtype = Dtype::parse(value["dtype"].as_str().context("missing tensor dtype")?)?;
            let shape = value["shape"]
                .as_array()
                .context("missing tensor shape")?
                .iter()
                .map(integer)
                .collect::<Result<Vec<_>>>()?;
            let elements = shape
                .iter()
                .try_fold(1usize, |n, &dim| n.checked_mul(dim))
                .context("tensor element count overflow")?;
            let byte_len = elements
                .checked_mul(dtype.bytes())
                .context("tensor byte count overflow")?;
            let offsets = value["data_offsets"]
                .as_array()
                .context("missing tensor offsets")?;
            ensure!(offsets.len() == 2, "expected two tensor offsets");
            let start = integer(&offsets[0])?;
            let end = integer(&offsets[1])?;
            ensure!(
                start <= end && end as u64 <= data_len && end - start == byte_len,
                "invalid safetensors extent for {name}"
            );
            spans.push((start, end));
            tensors.insert(
                name.clone(),
                Tensor {
                    dtype,
                    shape,
                    offset: start as u64,
                    byte_len,
                },
            );
        }
        spans.sort_unstable();
        let mut end = 0;
        for (start, next) in spans {
            ensure!(
                start == end,
                "safetensors data has a gap or overlapping tensors"
            );
            end = next;
        }
        ensure!(end as u64 == data_len, "unindexed safetensors data");
        Ok(Self {
            tensors,
            data_offset: header.len() as u64 + 8,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(value: Value, size: u64) -> Result<Shard> {
        Shard::parse(&serde_json::to_vec(&value)?, size)
    }

    #[test]
    fn accepts_mixed_precision_and_zero_sized_tensors() -> Result<()> {
        let shard = parse(
            json!({
                "__metadata__": {"format": "pt"},
                "empty": {"dtype":"F32","shape":[0,16],"data_offsets":[0,0]},
                "weight": {"dtype":"I32","shape":[2,4],"data_offsets":[0,32]},
                "scale": {"dtype":"BF16","shape":[2],"data_offsets":[32,36]}
            }),
            36,
        )?;
        assert_eq!(shard.tensors["weight"].shape, [2, 4]);
        assert_eq!(shard.tensors["scale"].dtype, Dtype::Bf16);
        Ok(())
    }

    #[test]
    fn rejects_invalid_extents_and_unknown_types() {
        for value in [
            json!({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}),
            json!({"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}),
            json!({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},
                "b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}),
            json!({"a":{"dtype":"F32","shape":[u64::MAX,4],"data_offsets":[0,8]}}),
            json!({"a":{"dtype":"F64","shape":[1],"data_offsets":[0,8]}}),
            json!({"__metadata__":{"format":false}}),
        ] {
            assert!(parse(value, 8).is_err());
        }
    }
}
