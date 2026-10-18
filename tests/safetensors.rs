use anyhow::Result;
use lia::{
    quant::gsq::Matrix,
    safetensors::{Archive, Dtype},
};

#[test]
#[ignore = "requires the pinned GSQ model directory in LIA_TEST_GSQ_MODEL"]
fn sharded_gsq_checkpoint_preserves_mixed_precision_layouts() -> Result<()> {
    let archive = Archive::open(std::env::var("LIA_TEST_GSQ_MODEL")?)?;
    let expert = Matrix::load(
        &archive,
        "model.language_model.layers.0.mlp.experts.0.gate_proj",
        2,
        128,
    )?;
    assert_eq!((expert.rows, expert.columns), (512, 2048));
    assert_eq!(expert.packed.len(), 512 * 2048 / 4);
    assert!(expert.value(511, 2047)?.is_finite());
    let attention = Matrix::load(
        &archive,
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        8,
        128,
    )?;
    assert_eq!((attention.rows, attention.columns), (8192, 2048));
    let output = Matrix::load(&archive, "lm_head", 8, 0)?;
    assert_eq!((output.rows, output.columns), (248320, 2048));
    assert_eq!(output.scales.len(), 248320);
    assert_eq!(
        archive
            .tensor("model.language_model.embed_tokens.weight")?
            .dtype,
        Dtype::Bf16
    );
    Ok(())
}
