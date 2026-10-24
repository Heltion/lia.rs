//! ONNX graph metadata for explicit architecture-specific Rust loaders.
use super::{
    Tensor,
    protobuf::{Message, Reader, Value, integers},
    tensor,
};
use anyhow::{Result, ensure};
use std::collections::BTreeMap;
#[derive(Clone)]
pub enum Attribute {
    Float(f32),
    Int(i64),
    String(String),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Tensor(Tensor),
    Unsupported,
}
pub struct Node {
    pub name: String,
    pub operation: String,
    pub domain: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: BTreeMap<String, Attribute>,
}
fn attribute(message: Message<'_>) -> Result<(String, Attribute)> {
    let mut reader = Reader::new(message);
    let mut name = String::new();
    let mut value = None;
    while let Some(field) = reader.next()? {
        let item = match field.number {
            1 => {
                name = field.value.text()?;
                continue;
            }
            2 => Attribute::Float(field.value.float()?),
            3 => Attribute::Int(field.value.integer()? as i64),
            4 => Attribute::String(field.value.text()?),
            5 => Attribute::Tensor(tensor::parse(field.value.message()?)?),
            7 => {
                let bytes = match field.value {
                    Value::Fixed32(v) => v.to_vec(),
                    Value::Bytes(m) => m.bytes.to_vec(),
                    _ => anyhow::bail!("invalid ONNX float attribute"),
                };
                ensure!(
                    bytes.len().is_multiple_of(4),
                    "misaligned ONNX float attribute"
                );
                let values = value.get_or_insert_with(|| Attribute::Floats(Vec::new()));
                let Attribute::Floats(values) = values else {
                    anyhow::bail!("mixed ONNX attribute payload")
                };
                values.extend(
                    bytes
                        .chunks_exact(4)
                        .map(|v| f32::from_le_bytes(v.try_into().unwrap())),
                );
                continue;
            }
            8 => {
                let values = value.get_or_insert_with(|| Attribute::Ints(Vec::new()));
                let Attribute::Ints(values) = values else {
                    anyhow::bail!("mixed ONNX attribute payload")
                };
                values.extend(integers(field.value)?.into_iter().map(|v| v as i64));
                continue;
            }
            _ => continue,
        };
        ensure!(
            value.replace(item).is_none(),
            "duplicate ONNX attribute payload"
        );
    }
    ensure!(!name.is_empty(), "unnamed ONNX attribute");
    Ok((name, value.unwrap_or(Attribute::Unsupported)))
}
fn node(message: Message<'_>) -> Result<Node> {
    let mut reader = Reader::new(message);
    let mut node = Node {
        name: String::new(),
        operation: String::new(),
        domain: String::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        attributes: BTreeMap::new(),
    };
    while let Some(field) = reader.next()? {
        match field.number {
            1 => {
                ensure!(node.inputs.len() < 4096, "too many ONNX node inputs");
                node.inputs.push(field.value.text()?);
            }
            2 => {
                ensure!(node.outputs.len() < 4096, "too many ONNX node outputs");
                node.outputs.push(field.value.text()?);
            }
            3 => node.name = field.value.text()?,
            4 => node.operation = field.value.text()?,
            5 => {
                let (name, value) = attribute(field.value.message()?)?;
                ensure!(
                    node.attributes.insert(name, value).is_none(),
                    "duplicate ONNX node attribute"
                );
            }
            7 => node.domain = field.value.text()?,
            _ => {}
        }
    }
    ensure!(
        !node.operation.is_empty() && !node.outputs.is_empty(),
        "incomplete ONNX node metadata"
    );
    Ok(node)
}
pub(super) struct Graph {
    pub tensors: BTreeMap<String, Tensor>,
    pub nodes: Vec<Node>,
}
pub(super) fn parse(message: Message<'_>) -> Result<Graph> {
    let mut reader = Reader::new(message);
    let mut tensors = BTreeMap::new();
    let mut nodes = Vec::new();
    while let Some(field) = reader.next()? {
        match field.number {
            1 => {
                ensure!(nodes.len() < 100000, "too many ONNX nodes");
                nodes.push(node(field.value.message()?)?);
            }
            5 => {
                ensure!(tensors.len() < 100000, "too many ONNX tensors");
                let t = tensor::parse(field.value.message()?)?;
                ensure!(
                    !t.name.is_empty() && tensors.insert(t.name.clone(), t).is_none(),
                    "unnamed or duplicate ONNX initializer"
                );
            }
            15 => anyhow::bail!("sparse ONNX initializers are unsupported"),
            _ => {}
        }
    }
    for node in &nodes {
        if node.operation == "Constant" && node.domain.is_empty() {
            ensure!(
                node.outputs.len() == 1,
                "invalid ONNX tensor Constant outputs"
            );
            let Some(Attribute::Tensor(t)) = node.attributes.get("value") else {
                anyhow::bail!("unsupported ONNX Constant representation")
            };
            ensure!(
                !node.outputs[0].is_empty()
                    && tensors.insert(node.outputs[0].clone(), t.clone()).is_none(),
                "duplicate ONNX Constant output"
            );
        }
    }
    Ok(Graph { tensors, nodes })
}
