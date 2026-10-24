#![cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};
use lia::inference::{Engine, PagedRow};

#[test]
#[ignore = "requires MiniCPM5 FP32 oracle, CUDA, and LIA_PREFILL_BATCH=128"]
fn sixty_four_requests_and_interleaved_prefill_match_independent_logits() -> Result<()> {
    let directory = std::path::PathBuf::from(std::env::var("LIA_TEST_MINICPM5_REFERENCE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(directory.join("info.json"))?)?;
    let tokens: Vec<u32> = serde_json::from_value(info["tokens"].clone())?;
    let bytes = std::fs::read(directory.join("logits.f32"))?;
    let reference: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")))
        .collect();
    let mut engine = Engine::load(&std::env::var("LIA_TEST_MINICPM5_MODEL")?, 2048)?;
    ensure!(
        engine.prefill_batch() == 128,
        "set LIA_PREFILL_BATCH=128 to validate bounded workspace"
    );
    let vocabulary = engine.vocabulary_size();
    ensure!(
        tokens.len() >= 2 && reference.len() >= 2 * vocabulary,
        "incomplete reference"
    );
    engine.configure_paging(64, 32)?;
    let tables: Vec<_> = (0..64).map(|sequence| vec![sequence]).collect();
    let outputs: Vec<_> = (0..64).collect();
    for position in 0..2 {
        let rows: Vec<_> = (0..64)
            .map(|sequence| PagedRow {
                token: tokens[position],
                position: position as u32,
                sequence,
            })
            .collect();
        let actual = engine.forward_paged(&rows, &tables, &outputs)?;
        verify(
            &actual,
            &reference[position * vocabulary..(position + 1) * vocabulary],
        )?;
    }
    engine.reset()?;
    let rows: Vec<_> = (0..2)
        .flat_map(|position| {
            let token = tokens[position];
            (0..64).map(move |sequence| PagedRow {
                token,
                position: position as u32,
                sequence,
            })
        })
        .collect();
    let outputs: Vec<_> = (64..128).collect();
    let actual = engine.forward_paged(&rows, &tables, &outputs)?;
    verify(
        &actual,
        reference
            .get(vocabulary..2 * vocabulary)
            .context("missing second oracle row")?,
    )?;
    Ok(())
}

fn verify(actual: &[f32], expected: &[f32]) -> Result<()> {
    ensure!(
        actual.len() == expected.len() * 64,
        "packed batch extent mismatch"
    );
    let mut maximum = 0_f32;
    for row in actual.chunks_exact(expected.len()) {
        for (&a, &b) in row.iter().zip(expected) {
            ensure!(a.is_finite(), "nonfinite batch logit");
            maximum = maximum.max((a - b).abs());
        }
    }
    println!("64-request logits maximum error={maximum}");
    ensure!(maximum < 0.025, "64-request logits differ from FP32 oracle");
    Ok(())
}
