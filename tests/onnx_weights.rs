use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
#[test]
#[ignore = "requires official MiniCPM-o ONNX weights and external canonical tensor oracle"]
fn embedded_onnx_weights_match_official_parser_byte_for_byte() -> Result<()> {
    let model = std::path::PathBuf::from(std::env::var("LIA_MINICPMO45_MODEL")?);
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_ONNX_ORACLE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    for file in info["files"].as_array().context("missing ONNX files")? {
        let filename = file["file"].as_str().context("missing ONNX filename")?;
        let archive =
            lia::onnx_weights::Archive::open(model.join("assets/token2wav").join(filename))?;
        let expected = file["tensors"]
            .as_array()
            .context("missing ONNX tensor inventory")?;
        ensure!(
            archive.tensors().len() == expected.len()
                && archive.nodes().len() == file["nodes"].as_u64().unwrap() as usize,
            "ONNX graph inventory mismatch"
        );
        let mut operations = std::collections::BTreeMap::<String, usize>::new();
        for node in archive.nodes() {
            *operations.entry(node.operation.clone()).or_default() += 1;
        }
        ensure!(
            serde_json::to_value(operations)? == file["operations"],
            "ONNX operator metadata mismatch"
        );
        let nodes = file["node_records"]
            .as_array()
            .context("missing ONNX node records")?;
        for (node, expected) in archive.nodes().iter().zip(nodes) {
            ensure!(
                node.name == expected["name"].as_str().unwrap()
                    && node.operation == expected["operation"].as_str().unwrap()
                    && node.domain == expected["domain"].as_str().unwrap(),
                "ONNX node identity mismatch"
            );
            ensure!(
                serde_json::to_value(&node.inputs)? == expected["inputs"]
                    && serde_json::to_value(&node.outputs)? == expected["outputs"],
                "ONNX graph connection mismatch"
            );
            let attrs = expected["attributes"].as_object().unwrap();
            ensure!(
                node.attributes.len() == attrs.len(),
                "ONNX attribute count mismatch"
            );
            for (name, attribute) in &node.attributes {
                check_attribute(&archive, attribute, &attrs[name])?;
            }
        }
        let mut bytes = 0;
        for entry in expected {
            let name = entry["name"].as_str().context("missing tensor name")?;
            let tensor = archive.tensor(name)?;
            let shape: Vec<usize> = serde_json::from_value(entry["shape"].clone())?;
            ensure!(
                tensor.shape == shape
                    && tensor.dtype == entry["dtype"].as_u64().unwrap() as u32
                    && tensor.name == entry["tensor_name"].as_str().unwrap(),
                "ONNX tensor metadata mismatch for {name}"
            );
            let data = archive.bytes(name)?;
            ensure!(
                data.len() == entry["bytes"].as_u64().unwrap() as usize,
                "ONNX tensor byte count mismatch"
            );
            ensure!(
                format!("{:x}", Sha256::digest(data)) == entry["sha256"].as_str().unwrap(),
                "ONNX tensor data mismatch for {name}"
            );
            bytes += data.len();
        }
        println!(
            "{filename}: tensors={}, nodes={}, verified_bytes={bytes}",
            expected.len(),
            archive.nodes().len()
        );
    }
    Ok(())
}

fn check_attribute(
    archive: &lia::onnx_weights::Archive,
    actual: &lia::onnx_weights::Attribute,
    expected: &serde_json::Value,
) -> Result<()> {
    use lia::onnx_weights::Attribute;
    match actual {
        Attribute::Float(v) => ensure!(
            expected["type"] == 1
                && v.to_bits() == (expected["value"].as_f64().unwrap() as f32).to_bits(),
            "ONNX Float32 attribute mismatch"
        ),
        Attribute::Int(v) => ensure!(
            expected["type"] == 2 && Some(*v) == expected["value"].as_i64(),
            "ONNX integer attribute mismatch"
        ),
        Attribute::String(v) => ensure!(
            expected["type"] == 3 && Some(v.as_str()) == expected["value"].as_str(),
            "ONNX string attribute mismatch"
        ),
        Attribute::Floats(v) => {
            let values: Vec<f32> = serde_json::from_value(expected["value"].clone())?;
            ensure!(
                expected["type"] == 6 && *v == values,
                "ONNX floating array mismatch"
            );
        }
        Attribute::Ints(v) => ensure!(
            expected["type"] == 7 && serde_json::to_value(v)? == expected["value"],
            "ONNX integer array mismatch"
        ),
        Attribute::Tensor(v) => {
            let shape: Vec<usize> = serde_json::from_value(expected["shape"].clone())?;
            ensure!(
                expected["type"] == 4
                    && v.shape == shape
                    && Some(u64::from(v.dtype)) == expected["dtype"].as_u64(),
                "ONNX tensor attribute metadata mismatch"
            );
            ensure!(
                format!("{:x}", Sha256::digest(archive.data(v)?))
                    == expected["sha256"].as_str().unwrap(),
                "ONNX tensor attribute bytes mismatch"
            );
        }
        Attribute::Unsupported => anyhow::bail!("unsupported fixture attribute"),
    }
    Ok(())
}
