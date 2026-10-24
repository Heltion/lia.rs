use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};

#[test]
#[ignore = "requires official external token2wav checkpoints and Torch byte oracle"]
fn codec_weights_match_official_torch_reader() -> Result<()> {
    let path = std::env::var("LIA_TORCH_READER_ORACLE")?;
    let info: serde_json::Value = serde_json::from_reader(std::fs::File::open(path)?)?;
    let mut total = 0;
    for case in info["cases"].as_array().context("missing cases")? {
        let path = case["path"].as_str().context("missing checkpoint path")?;
        let archive = lia::torch_weights::Archive::open(path)?;
        let entries = case["tensors"]
            .as_array()
            .context("missing tensor entries")?;
        ensure!(
            archive.tensors().len() == entries.len(),
            "incorrect archive tensor count"
        );
        for entry in entries {
            let name = entry["name"].as_str().unwrap();
            let tensor = archive.tensor(name)?;
            let shape: Vec<usize> = serde_json::from_value(entry["shape"].clone())?;
            let stride: Vec<usize> = serde_json::from_value(entry["stride"].clone())?;
            ensure!(
                tensor.shape == shape && tensor.stride == stride,
                "tensor layout mismatch for {name}"
            );
            let bytes = archive.read(name)?;
            let actual = format!("{:x}", Sha256::digest(&bytes));
            ensure!(
                Some(actual.as_str()) == entry["sha256"].as_str(),
                "weight byte mismatch for {name}"
            );
        }
        total += entries.len();
        println!(
            "{path}: {} tensor shapes, strides and raw-byte SHA256 values match",
            entries.len()
        );
    }
    let root = std::path::PathBuf::from(info["fixtures"].as_str().unwrap());
    let small = lia::torch_weights::Archive::open(root.join("small.pt"))?;
    let values = small.read("sample")?;
    let values: Vec<f32> = values
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    ensure!(
        values == vec![1., 2., 3., 4.],
        "small weight fixture mismatch"
    );
    let corrupt = lia::torch_weights::Archive::open(root.join("corrupt.pt"))?;
    ensure!(
        corrupt.read("sample").is_err(),
        "corrupt storage CRC was accepted"
    );
    ensure!(
        lia::torch_weights::Archive::open(root.join("unsupported-global.pt")).is_err(),
        "executable GLOBAL was accepted"
    );
    ensure!(
        lia::torch_weights::Archive::open(root.join("noncontiguous.pt")).is_err(),
        "unsupported noncontiguous view was accepted"
    );
    println!("{total} official tensors and 4 independent storage fixtures checked");
    Ok(())
}
