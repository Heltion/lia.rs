#![cfg(feature = "cuda")]
use anyhow::{Context, Result};
use lia::{
    inference::{Engine, PagedRow},
    sampling::Sampling,
};

#[test]
#[ignore = "requires MiniCPM5 checkpoint and CUDA device"]
fn batched_sparse_samples_match_complete_logit_distributions() -> Result<()> {
    let mut engine = Engine::load(&std::env::var("LIA_TEST_MINICPM5_MODEL")?, 256)?;
    engine.configure_paging(8, 32)?;
    let tokens = [0, 36417, 1782, 22, 75828, 68572, 1398, 35];
    let tables: Vec<_> = (0..8).map(|i| vec![i]).collect();
    for batch in [1, 2, 3, 4, 5, 8] {
        let rows: Vec<_> = (0..batch)
            .map(|sequence| PagedRow {
                token: tokens[sequence],
                position: 0,
                sequence: sequence as u32,
            })
            .collect();
        let outputs: Vec<_> = (0..batch as u32).collect();
        let logits = engine.forward_paged(&rows, &tables, &outputs)?;
        let mut expected = Vec::new();
        let mut samplers = Vec::new();
        for (index, row) in logits.chunks_exact(engine.vocabulary_size()).enumerate() {
            let make = || -> Result<Option<Sampling>> {
                if index == 0 {
                    return Ok(None);
                }
                let (top_k, top_p) = match index % 4 {
                    1 => (0, 0.95),
                    2 => (20, 0.95),
                    3 => (0, 1.0),
                    _ => (0, 0.5),
                };
                Ok(Some(
                    Sampling::new(0.7 + index as f64 * 0.1, top_k, top_p, 42 + index as u64)?
                        .with_position_gumbel(),
                ))
            };
            let next = if let Some(mut sampling) = make()? {
                let distribution = sampling
                    .distribution(row.iter().copied().enumerate().map(|(i, v)| (i as u32, v)))?;
                sampling.draw(&distribution)
            } else {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .context("empty logits")?
                    .0 as u32
            };
            expected.push(next);
            samplers.push(make()?);
        }
        let actual = engine.forward_paged_sampled(&rows, &tables, &outputs, &mut samplers)?;
        assert_eq!(
            actual, expected,
            "batch {batch} sparse sampler changed the seeded choices"
        );
    }
    assert!(
        engine.nucleus_fallbacks() > 0,
        "unrestricted top_p=1 did not exercise full fallback"
    );
    Ok(())
}
