use anyhow::{Context, Result};
use lia::{models::minicpm_o45::Config, quant::awq::Matrix, safetensors::Archive};

#[test]
#[ignore = "requires official MiniCPM-o 4.5 AWQ checkpoint outside Git"]
fn official_awq_checkpoint_contract() -> Result<()> {
    let directory = std::env::var("LIA_MINICPMO45_MODEL")
        .context("set LIA_MINICPMO45_MODEL to the official AWQ checkpoint")?;
    let config = Config::open(&directory)?;
    let archive = Archive::open(&directory)?;
    config.validate_text_weights(&archive)?;
    for layer in [0, 17, 35] {
        for projection in ["self_attn.q_proj", "self_attn.k_proj", "mlp.down_proj"] {
            let name = format!("llm.model.layers.{layer}.{projection}");
            let matrix = Matrix::load(&archive, &name, config.group_size)?;
            for row in [0, 7, matrix.rows() - 1] {
                for column in [0, 127, 128, matrix.columns() - 1] {
                    assert!(matrix.value(row, column)?.is_finite(), "{name}");
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires official AWQ checkpoint and an external independent decoding oracle"]
fn official_awq_values_match_oracle() -> Result<()> {
    let directory = std::env::var("LIA_MINICPMO45_MODEL")
        .context("set LIA_MINICPMO45_MODEL to the official AWQ checkpoint")?;
    let path = std::env::var("LIA_AWQ_ORACLE").context("set LIA_AWQ_ORACLE")?;
    let oracle: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(path).context("opening independent AWQ decoding oracle")?,
    )
    .context("reading independent AWQ decoding oracle")?;
    let config = Config::open(&directory)?;
    let archive = Archive::open(directory)?;
    let cases = oracle.as_array().context("expected AWQ oracle array")?;
    anyhow::ensure!(!cases.is_empty(), "empty AWQ oracle");
    let mut count = 0;
    for case in cases {
        let prefix = case["prefix"]
            .as_str()
            .context("missing AWQ oracle tensor")?;
        let matrix = Matrix::load(&archive, prefix, config.group_size)?;
        let samples = case["samples"]
            .as_array()
            .context("missing AWQ oracle samples")?;
        anyhow::ensure!(!samples.is_empty(), "empty AWQ tensor samples");
        for sample in samples {
            let row = usize::try_from(sample["row"].as_u64().context("missing row")?)?;
            let column = usize::try_from(sample["column"].as_u64().context("missing column")?)?;
            let expected = sample["value"]
                .as_f64()
                .context("missing reference weight")? as f32;
            assert_eq!(
                matrix.value(row, column)?.to_bits(),
                expected.to_bits(),
                "{prefix} at row {row}, column {column}"
            );
            count += 1;
        }
    }
    println!(
        "{} AWQ matrices, {count} sampled values match exactly",
        cases.len()
    );
    Ok(())
}
