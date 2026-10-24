//! Fixed S3 v2 25 Hz ONNX tensor contracts; execution stays in native Rust kernels.
use crate::{
    models::parameters::{Parameter, Part},
    onnx_weights::{Archive, Attribute, Node},
};
use anyhow::{Context, Result, ensure};
fn node<'a>(archive: &'a Archive, name: &str, op: &str) -> Result<&'a Node> {
    let mut nodes = archive.nodes().iter().filter(|n| n.name == name);
    let found = nodes
        .next()
        .with_context(|| format!("missing S3 node {name}"))?;
    ensure!(
        nodes.next().is_none() && found.operation == op && found.domain.is_empty(),
        "unexpected S3 node {name}"
    );
    Ok(found)
}
fn tensor(
    out: &mut Vec<Parameter>,
    archive: &Archive,
    target: String,
    source: &str,
    shape: Vec<usize>,
    matrix: bool,
    transpose: bool,
) -> Result<()> {
    let value = archive.tensor(source)?;
    ensure!(
        value.shape == shape && value.dtype == 1,
        "S3 tensor shape/dtype mismatch: {source}"
    );
    let assembled = if matrix {
        if transpose {
            vec![shape[1], shape[0]]
        } else {
            vec![shape[0], shape[1..].iter().product()]
        }
    } else {
        shape.clone()
    };
    out.push(Parameter {
        target: format!("s3.{target}"),
        parts: vec![Part::new(source, shape)],
        shape: assembled,
        matrix,
        transpose,
    });
    Ok(())
}
fn linear(
    out: &mut Vec<Parameter>,
    archive: &Archive,
    scope: &str,
    target: &str,
    inputs: usize,
    outputs: usize,
    bias: bool,
) -> Result<()> {
    let n = node(archive, &format!("{scope}/MatMul"), "MatMul")?;
    ensure!(n.inputs.len() == 2, "invalid S3 linear inputs");
    tensor(
        out,
        archive,
        format!("{target}.weight"),
        &n.inputs[1],
        vec![inputs, outputs],
        true,
        true,
    )?;
    if bias {
        let add = node(archive, &format!("{scope}/Add"), "Add")?;
        ensure!(
            add.inputs.len() == 2 && add.inputs.contains(&n.outputs[0]),
            "invalid S3 linear bias connection"
        );
        let source = add
            .inputs
            .iter()
            .find(|i| **i != n.outputs[0])
            .context("missing S3 bias")?;
        tensor(
            out,
            archive,
            format!("{target}.bias"),
            source,
            vec![outputs],
            false,
            false,
        )?;
    }
    Ok(())
}
fn conv(
    out: &mut Vec<Parameter>,
    archive: &Archive,
    scope: &str,
    target: &str,
    shape: Vec<usize>,
    stride: i64,
    group: i64,
    bias: bool,
) -> Result<()> {
    let n = node(archive, &format!("{scope}/Conv"), "Conv")?;
    ensure!(
        n.inputs.len() == if bias { 3 } else { 2 },
        "invalid S3 convolution inputs"
    );
    for (name, value) in [
        ("strides", vec![stride]),
        (
            "pads",
            vec![if group == 1 { (shape[2] / 2) as i64 } else { 0 }; 2],
        ),
        ("dilations", vec![1]),
        ("kernel_shape", vec![shape[2] as i64]),
    ] {
        ensure!(
            matches!(n.attributes.get(name),Some(Attribute::Ints(v))if *v==value),
            "unexpected S3 convolution {scope} {name}"
        );
    }
    ensure!(
        matches!(n.attributes.get("group"),Some(Attribute::Int(v))if *v==group),
        "unexpected S3 convolution groups"
    );
    tensor(
        out,
        archive,
        format!("{target}.weight"),
        &n.inputs[1],
        shape.clone(),
        group == 1,
        false,
    )?;
    if bias {
        tensor(
            out,
            archive,
            format!("{target}.bias"),
            &n.inputs[2],
            vec![shape[0]],
            false,
            false,
        )?;
    }
    Ok(())
}
pub(crate) fn plan(archive: &Archive) -> Result<Vec<Parameter>> {
    let mut out = Vec::new();
    for (name, inputs) in [("conv1", 128), ("conv2", 1280)] {
        conv(
            &mut out,
            archive,
            &format!("/{name}"),
            name,
            vec![1280, inputs, 3],
            2,
            1,
            true,
        )?;
    }
    for layer in 0..6 {
        let base = format!("/blocks.{layer}");
        let target = format!("blk.{layer}");
        for (name, short) in [("attn_ln", "ln1"), ("mlp_ln", "ln2")] {
            let n = node(
                archive,
                &format!("{base}/{name}/LayerNormalization"),
                "LayerNormalization",
            )?;
            ensure!(
                n.inputs.len() == 3
                    && matches!(n.attributes.get("epsilon"),Some(Attribute::Float(v))if *v==1e-5)
                    && matches!(n.attributes.get("axis"), Some(Attribute::Int(-1))),
                "unsupported S3 layer normalization"
            );
            for (suffix, source) in ["weight", "bias"].into_iter().zip(&n.inputs[1..]) {
                tensor(
                    &mut out,
                    archive,
                    format!("{target}.{short}.{suffix}"),
                    source,
                    vec![1280],
                    false,
                    false,
                )?;
            }
        }
        for (name, short, bias) in [
            ("query", "q", true),
            ("key", "k", false),
            ("value", "v", true),
            ("out", "out", true),
        ] {
            linear(
                &mut out,
                archive,
                &format!("{base}/attn/{name}"),
                &format!("{target}.{short}"),
                1280,
                1280,
                bias,
            )?;
        }
        conv(
            &mut out,
            archive,
            &format!("{base}/attn/fsmn_block"),
            &format!("{target}.fsmn"),
            vec![1280, 1, 31],
            1,
            1280,
            false,
        )?;
        for (index, short, inputs, outputs) in [(0, "up", 1280, 5120), (2, "down", 5120, 1280)] {
            linear(
                &mut out,
                archive,
                &format!("{base}/mlp/mlp.{index}"),
                &format!("{target}.{short}"),
                inputs,
                outputs,
                true,
            )?;
        }
    }
    linear(
        &mut out,
        archive,
        "/quantizer/project_in",
        "quantizer",
        1280,
        8,
        true,
    )?;
    ensure!(out.len() == 102, "incomplete S3 weight plan");
    Ok(out)
}
