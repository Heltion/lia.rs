use anyhow::{Context, Result, ensure};
#[test]
#[ignore = "requires official flow mel fixtures outside Git"]
fn flow_mel_matches_official_reference() -> Result<()> {
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_FLOW_MEL_ORACLE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let processor = lia::models::minicpm_o45::flow_mel::Processor::default();
    let read = |name: &str| -> Result<Vec<f32>> {
        let bytes = std::fs::read(root.join(name))?;
        ensure!(bytes.len().is_multiple_of(4), "invalid Float32 fixture");
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    };
    let mut passed = true;
    for case in info["cases"].as_array().context("missing flow mel cases")? {
        let samples = read(case["input"].as_str().context("missing PCM fixture")?)?;
        let expected = read(case["mel"].as_str().context("missing mel fixture")?)?;
        let actual = processor.extract(&samples)?;
        ensure!(
            actual.frames == case["frames"].as_u64().context("missing frame count")? as usize
                && actual.values.len() == expected.len(),
            "flow mel extent mismatch"
        );
        std::fs::write(
            root.join(format!(
                "{}.native.f32",
                case["name"].as_str().context("missing case name")?
            )),
            actual
                .values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let maximum = actual
            .values
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f32::max);
        let rms = (actual
            .values
            .iter()
            .zip(&expected)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum::<f64>()
            / expected.len() as f64)
            .sqrt();
        println!("{}: max_abs={maximum}, rms={rms}", case["name"]);
        passed &= maximum < 0.0005 && rms < 0.00005;
    }
    ensure!(passed, "flow mel official numerical gate failed");
    Ok(())
}
