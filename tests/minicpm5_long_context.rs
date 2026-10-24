#![cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};
use lia::inference::{Engine, PagedRow};

#[test]
#[ignore = "strict ideal-FP32 16K audit; known F16 deviation; requires external oracle and CUDA"]
fn long_context_prefill_matches_independent_fp32_oracle() -> Result<()> {
    let directory = std::path::PathBuf::from(
        std::env::var("LIA_TEST_MINICPM5_RETRIEVAL_REFERENCE")
            .context("set the external long-context oracle directory")?,
    );
    let info: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(directory.join("info.json")).context("opening long-context oracle")?,
    )?;
    let tokens: Vec<u32> = serde_json::from_value(info["tokens"].clone())?;
    let bytes = std::fs::read(directory.join("logits.f32"))?;
    ensure!(bytes.len().is_multiple_of(4), "truncated FP32 oracle");
    let expected: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")))
        .collect();
    ensure!(
        tokens.len() > 8192,
        "oracle must exercise long-context attention"
    );
    let model = std::env::var("LIA_TEST_MINICPM5_MODEL")?;
    for count in [1, 4] {
        let capacity = tokens.len().div_ceil(32) * 32;
        let mut engine = Engine::load(&model, capacity * count)?;
        ensure!(
            expected.len() == engine.vocabulary_size(),
            "oracle shape mismatch"
        );
        engine.configure_paging(count, capacity)?;
        let pages = capacity / 32;
        let tables: Vec<Vec<u32>> = (0..count)
            .map(|sequence| {
                (0..pages)
                    .rev()
                    .map(|page| (sequence * pages + page) as u32)
                    .collect()
            })
            .collect();
        let submitted_rows = std::env::var("LIA_TEST_PREFILL_ROWS")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()
            .context("parsing diagnostic prefill rows")?
            .unwrap_or(engine.prefill_batch());
        ensure!(
            submitted_rows <= engine.prefill_batch(),
            "test exceeds workspace"
        );
        let chunk = submitted_rows / count;
        ensure!(chunk > 0, "invalid test prefill chunk");
        for begin in (0..tokens.len()).step_by(chunk) {
            let end = (begin + chunk).min(tokens.len());
            let mut rows = Vec::new();
            let mut outputs = Vec::new();
            for sequence in 0..count {
                for (position, &token) in tokens.iter().enumerate().take(end).skip(begin) {
                    rows.push(PagedRow {
                        token,
                        position: position as u32,
                        sequence: sequence as u32,
                    });
                }
                if end == tokens.len() {
                    outputs.push((rows.len() - 1) as u32);
                }
            }
            let actual = engine.forward_paged(&rows, &tables, &outputs)?;
            if end == tokens.len() {
                ensure!(
                    actual.len() == count * expected.len(),
                    "logit extent mismatch"
                );
                for (sequence, row) in actual.chunks_exact(expected.len()).enumerate() {
                    let mut maximum = 0_f32;
                    for (&a, &b) in row.iter().zip(&expected) {
                        ensure!(
                            a.is_finite() && b.is_finite(),
                            "nonfinite long-context logit"
                        );
                        maximum = maximum.max((a - b).abs());
                    }
                    println!("long context count={count} sequence={sequence} max_abs={maximum}");
                    ensure!(
                        maximum < 0.025,
                        "long-context logits differ from FP32 oracle"
                    );
                }
            }
        }
    }
    Ok(())
}
