//! Independent reader for the little-endian GGUF v2/v3 container.
use anyhow::Context;
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, Read, Seek},
    path::Path,
};

pub enum Value {
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Array { kind: u32, values: Vec<Value> },
}
#[derive(Debug)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub kind: u32,
    pub offset: u64,
    pub byte_len: u64,
}
pub struct Model {
    pub version: u32,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<Tensor>,
    pub data_offset: u64,
}
fn invalid(message: &str) -> anyhow::Error {
    anyhow::anyhow!("{message}")
}
struct Reader {
    inner: BufReader<File>,
    size: u64,
}
impl Reader {
    fn bytes<const N: usize>(&mut self) -> anyhow::Result<[u8; N]> {
        let mut b = [0; N];
        self.inner.read_exact(&mut b)?;
        Ok(b)
    }
    fn u32(&mut self) -> anyhow::Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> anyhow::Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn string(&mut self) -> anyhow::Result<String> {
        let len = self.u64()?;
        if len > 64 * 1024 * 1024 || len > self.size.saturating_sub(self.inner.stream_position()?) {
            return Err(invalid("invalid string length"));
        }
        let mut b = vec![0; len as usize];
        self.inner.read_exact(&mut b)?;
        String::from_utf8(b).map_err(|_| invalid("invalid UTF-8"))
    }
    fn value(&mut self, kind: u32, depth: usize) -> anyhow::Result<Value> {
        if depth > 8 {
            return Err(invalid("metadata nesting too deep"));
        }
        Ok(match kind {
            0 => Value::Unsigned(self.bytes::<1>()?[0] as u64),
            1 => Value::Signed(i8::from_le_bytes(self.bytes()?) as i64),
            2 => Value::Unsigned(u16::from_le_bytes(self.bytes()?) as u64),
            3 => Value::Signed(i16::from_le_bytes(self.bytes()?) as i64),
            4 => Value::Unsigned(self.u32()? as u64),
            5 => Value::Signed(i32::from_le_bytes(self.bytes()?) as i64),
            6 => Value::Float(f32::from_le_bytes(self.bytes()?) as f64),
            7 => Value::Bool(match self.bytes::<1>()?[0] {
                0 => false,
                1 => true,
                _ => return Err(invalid("invalid boolean")),
            }),
            8 => Value::String(self.string()?),
            9 => {
                let kind = self.u32()?;
                let len = self.u64()?;
                if len > self.size {
                    return Err(invalid("invalid array length"));
                }
                let mut values = Vec::new();
                for _ in 0..len {
                    values.push(self.value(kind, depth + 1)?);
                }
                Value::Array { kind, values }
            }
            10 => Value::Unsigned(self.u64()?),
            11 => Value::Signed(i64::from_le_bytes(self.bytes()?)),
            12 => Value::Float(f64::from_le_bytes(self.bytes()?)),
            _ => return Err(invalid("unknown metadata type")),
        })
    }
}
impl Model {
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("opening GGUF {}", path.as_ref().display()))?;
        let size = file.metadata()?.len();
        let mut r = Reader {
            inner: BufReader::new(file),
            size,
        };
        if r.bytes::<4>()? != *b"GGUF" {
            return Err(invalid("not a little-endian GGUF file"));
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            return Err(invalid("unsupported GGUF version"));
        }
        let count = r.u64()?;
        let pairs = r.u64()?;
        if count > size / 24 || pairs > size / 12 {
            return Err(invalid("invalid container counts"));
        }
        let mut metadata = BTreeMap::new();
        for _ in 0..pairs {
            let key = r.string()?;
            let kind = r.u32()?;
            let value = r.value(kind, 0)?;
            if metadata.insert(key, value).is_some() {
                return Err(invalid("duplicate metadata key"));
            }
        }
        let alignment = match metadata.get("general.alignment") {
            Some(Value::Unsigned(x)) => *x,
            None => 32,
            _ => return Err(invalid("invalid alignment type")),
        };
        if !alignment.is_power_of_two() || alignment > 1024 * 1024 {
            return Err(invalid("invalid alignment"));
        }
        let mut tensors = Vec::new();
        let mut names = std::collections::BTreeSet::new();
        for _ in 0..count {
            let name = r.string()?;
            if !names.insert(name.clone()) {
                return Err(invalid("duplicate tensor name"));
            }
            let dims = r.u32()?;
            if !(1..=4).contains(&dims) {
                return Err(invalid("invalid tensor dimensions"));
            }
            let mut shape = Vec::new();
            let mut elements = 1u64;
            for _ in 0..dims {
                let dim = r.u64()?;
                if dim == 0 {
                    return Err(invalid("zero dimension"));
                }
                elements = elements
                    .checked_mul(dim)
                    .ok_or_else(|| invalid("tensor size overflow"))?;
                shape.push(dim);
            }
            let kind = r.u32()?;
            let offset = r.u64()?;
            if offset % alignment != 0 {
                return Err(invalid("unaligned tensor"));
            }
            let (block, bytes) =
                layout(kind).ok_or_else(|| invalid("unsupported tensor storage type"))?;
            if shape[0] % block != 0 {
                return Err(invalid("tensor row is not block aligned"));
            }
            let byte_len = (elements / block)
                .checked_mul(bytes)
                .ok_or_else(|| invalid("tensor byte size overflow"))?;
            tensors.push(Tensor {
                name,
                shape,
                kind,
                offset,
                byte_len,
            });
        }
        let position = r.inner.stream_position()?;
        let data_offset = position
            .checked_add(alignment - 1)
            .ok_or_else(|| invalid("offset overflow"))?
            & !(alignment - 1);
        let mut spans = Vec::new();
        for t in &tensors {
            let start = data_offset
                .checked_add(t.offset)
                .ok_or_else(|| invalid("tensor offset overflow"))?;
            let end = start
                .checked_add(t.byte_len)
                .ok_or_else(|| invalid("tensor extent overflow"))?;
            if end > size {
                return Err(invalid("truncated tensor data"));
            }
            spans.push((start, end));
        }
        spans.sort_unstable();
        if spans.windows(2).any(|w| w[0].1 > w[1].0) {
            return Err(invalid("overlapping tensors"));
        }
        Ok(Self {
            version,
            metadata,
            tensors,
            data_offset,
        })
    }
}

impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsigned(x) => write!(f, "{x}"),
            Self::Signed(x) => write!(f, "{x}"),
            Self::Float(x) => write!(f, "{x}"),
            Self::Bool(x) => write!(f, "{x}"),
            Self::String(x) => write!(f, "{x:?}"),
            Self::Array { kind, values } => write!(f, "Array(type={kind}, len={})", values.len()),
        }
    }
}

/// (elements, bytes) for one GGML storage block. Unknown encodings fail closed.
pub fn layout(kind: u32) -> Option<(u64, u64)> {
    Some(match kind {
        0 => (1, 4),
        1 | 30 => (1, 2),
        2 => (32, 18),
        3 => (32, 20),
        6 => (32, 22),
        7 => (32, 24),
        8 => (32, 34),
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        16 => (256, 66),
        17 => (256, 74),
        18 => (256, 98),
        19 => (256, 50),
        20 => (32, 18),
        21 => (256, 110),
        22 => (256, 82),
        23 => (256, 136),
        24 => (1, 1),
        25 => (1, 2),
        26 => (1, 4),
        27 | 28 => (1, 8),
        29 => (256, 56),
        _ => return None,
    })
}

#[cfg(test)]
mod tests;
