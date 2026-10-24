//! Assemble CAM++ tensors by ONNX module scope, with complete architecture validation.
use crate::{
    models::parameters::{Parameter, Part},
    onnx_weights::{Archive, Attribute, Node},
};
use anyhow::{Result, ensure};
mod schema;
use schema::Spec;
fn scope(node: &Node) -> String {
    let mut parts: Vec<String> = Vec::new();
    for part in node
        .name
        .split('/')
        .filter(|s| !s.is_empty() && !["Conv", "BatchNormalization"].contains(s))
    {
        let suffix = parts
            .last()
            .and_then(|previous| part.strip_prefix(&format!("{previous}.")));
        parts.push(suffix.unwrap_or(part).to_owned());
    }
    parts.join(".")
}
fn ints(node: &Node, name: &str, expected: &[i64]) -> Result<()> {
    ensure!(
        matches!(node.attributes.get(name),Some(Attribute::Ints(v))if v==expected),
        "unexpected speaker {} {name}",
        node.name
    );
    Ok(())
}
pub(crate) fn plan(archive: &Archive) -> Result<Vec<Parameter>> {
    let mut expected = schema::expected();
    let mut plan = Vec::new();
    for node in archive
        .nodes()
        .iter()
        .filter(|n| ["Conv", "BatchNormalization"].contains(&n.operation.as_str()))
    {
        ensure!(
            node.domain.is_empty(),
            "unsupported speaker operator domain"
        );
        let name = scope(node);
        let spec = expected
            .remove(&name)
            .ok_or_else(|| anyhow::anyhow!("unexpected speaker module {name}"))?;
        match spec {
            Spec::Conv {
                shape,
                stride,
                pads,
                dilation,
                bias,
            } => {
                ensure!(
                    node.operation == "Conv" && node.inputs.len() == if bias { 3 } else { 2 },
                    "unexpected speaker convolution {name}"
                );
                ints(node, "strides", &stride)?;
                ints(node, "pads", &pads)?;
                ints(node, "dilations", &dilation)?;
                ints(
                    node,
                    "kernel_shape",
                    &shape[2..].iter().map(|&v| v as i64).collect::<Vec<_>>(),
                )?;
                ensure!(
                    matches!(node.attributes.get("group"), Some(Attribute::Int(1))),
                    "grouped speaker convolution unsupported"
                );
                let source = &node.inputs[1];
                ensure!(
                    archive.tensor(source)?.shape == shape && archive.tensor(source)?.dtype == 1,
                    "speaker convolution shape/dtype mismatch"
                );
                plan.push(Parameter {
                    target: format!("speaker.{name}.weight"),
                    parts: vec![Part::new(source, shape.clone())],
                    shape: vec![shape[0], shape[1..].iter().product()],
                    matrix: true,
                    transpose: false,
                });
                if bias {
                    let source = &node.inputs[2];
                    ensure!(
                        archive.tensor(source)?.shape == [shape[0]]
                            && archive.tensor(source)?.dtype == 1,
                        "speaker bias shape/dtype mismatch"
                    );
                    plan.push(Parameter::floats(
                        format!("speaker.{name}.bias"),
                        source,
                        vec![shape[0]],
                    ));
                }
            }
            Spec::Norm(channels) => {
                ensure!(
                    node.operation == "BatchNormalization" && node.inputs.len() == 5,
                    "invalid speaker batch normalization"
                );
                ensure!(
                    matches!(node.attributes.get("epsilon"),Some(Attribute::Float(v))if *v==1e-5)
                        && matches!(
                            node.attributes.get("training_mode"),
                            Some(Attribute::Int(0))
                        ),
                    "unsupported speaker normalization mode"
                );
                for (slot, source) in ["weight", "bias", "mean", "variance"]
                    .into_iter()
                    .zip(&node.inputs[1..])
                {
                    ensure!(
                        archive.tensor(source)?.shape == [channels]
                            && archive.tensor(source)?.dtype == 1,
                        "speaker normalization parameter mismatch"
                    );
                    plan.push(Parameter::floats(
                        format!("speaker.{name}.{slot}"),
                        source,
                        vec![channels],
                    ));
                }
            }
        }
    }
    ensure!(
        expected.is_empty(),
        "missing speaker modules: {:?}",
        expected.keys().collect::<Vec<_>>()
    );
    Ok(plan)
}
