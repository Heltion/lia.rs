//! Read embedded dense ONNX tensors and graph metadata, without an inference runtime.
use anyhow::{Context, Result, ensure};
use std::{collections::BTreeMap, path::Path};
mod graph;
mod protobuf;
mod tensor;
pub use graph::{Attribute, Node};
use protobuf::{Message, Reader};
pub use tensor::Tensor;
pub struct Archive {
    bytes: Vec<u8>,
    tensors: BTreeMap<String, Tensor>,
    nodes: Vec<Node>,
}
impl Archive {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let length = std::fs::metadata(path)
            .with_context(|| format!("stat ONNX weights {}", path.display()))?
            .len();
        ensure!(
            length <= 2 * 1024 * 1024 * 1024,
            "ONNX file exceeds container size limit"
        );
        let bytes =
            std::fs::read(path).with_context(|| format!("read ONNX weights {}", path.display()))?;
        Self::parse(bytes).with_context(|| format!("parse ONNX weights {}", path.display()))
    }
    fn parse(bytes: Vec<u8>) -> Result<Self> {
        let mut reader = Reader::new(Message {
            bytes: &bytes,
            offset: 0,
        });
        let mut graph = None;
        while let Some(field) = reader.next()? {
            if field.number == 7 {
                ensure!(graph.is_none(), "duplicate ONNX model graph");
                graph = Some(graph::parse(field.value.message()?)?);
            }
        }
        let graph = graph.context("missing ONNX model graph")?;
        Ok(Self {
            bytes,
            tensors: graph.tensors,
            nodes: graph.nodes,
        })
    }
    pub fn tensors(&self) -> &BTreeMap<String, Tensor> {
        &self.tensors
    }
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .with_context(|| format!("missing ONNX tensor {name}"))
    }
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        self.data(self.tensor(name)?)
    }
    pub fn data<'a>(&'a self, tensor: &'a Tensor) -> Result<&'a [u8]> {
        match &tensor.data {
            tensor::Data::Raw(range) => self
                .bytes
                .get(range.clone())
                .context("ONNX tensor lies outside container"),
            tensor::Data::Owned(bytes) => Ok(bytes),
        }
    }
    pub fn floats(&self, name: &str) -> Result<Vec<f32>> {
        ensure!(
            self.tensor(name)?.dtype == 1,
            "ONNX tensor {name} is not Float32"
        );
        Ok(self
            .bytes(name)?
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_containers_are_rejected() {
        assert!(Archive::parse(vec![]).is_err());
        assert!(Archive::parse(vec![0x3a, 9]).is_err());
        assert!(Archive::parse(vec![0x3a, 0, 0x3a, 0]).is_err());
    }
}
