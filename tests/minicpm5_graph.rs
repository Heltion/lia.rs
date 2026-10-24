#![cfg(feature = "cuda")]
use anyhow::{Result, ensure};
use lia::inference::{Engine, PagedRow};

#[test]
#[ignore = "requires MiniCPM5 checkpoint and a CUDA device"]
fn graph_partition_span_change_keeps_all_causal_keys() -> Result<()> {
    let model = std::env::var("LIA_TEST_MINICPM5_MODEL")?;
    let tokens: Vec<_> = std::iter::once(0)
        .chain([36417, 1782, 22].into_iter().cycle())
        .take(2050)
        .collect();
    let mut engine = Engine::load(&model, 4096)?;
    let mut processed = 0;
    while processed < 2046 {
        let count = engine.prefill_chunk(2046 - processed)?;
        engine.forward_many(&tokens[processed..processed + count], false)?;
        processed += count;
    }
    let mut expected = Vec::new();
    for &token in &tokens[2046..] {
        expected.push(engine.forward(token)?);
    }
    engine.reset()?;
    engine.configure_paging(1, 4096)?;
    let tables = [(0..128).rev().collect::<Vec<_>>()];
    let mut processed = 0;
    while processed < 2046 {
        let count = engine.prefill_chunk(2046 - processed)?;
        let rows: Vec<_> = (processed..processed + count)
            .map(|position| PagedRow {
                token: tokens[position],
                position: position as u32,
                sequence: 0,
            })
            .collect();
        engine.forward_paged(&rows, &tables, &[])?;
        processed += count;
    }
    for (index, position) in (2046..2050).enumerate() {
        let before = engine.cuda_graph_stats();
        let row = PagedRow {
            token: tokens[position],
            position: position as u32,
            sequence: 0,
        };
        let actual = engine.forward_paged(&[row], &tables, &[0])?;
        ensure!(
            actual.len() == expected[index].len(),
            "logit extent mismatch"
        );
        let mut maximum = 0_f32;
        for (&a, &b) in actual.iter().zip(&expected[index]) {
            ensure!(
                a.is_finite() && b.is_finite(),
                "nonfinite graph boundary logit"
            );
            maximum = maximum.max((a - b).abs());
        }
        println!("paged graph position {position}: max_abs={maximum}");
        ensure!(maximum < 0.025, "graph omitted or mixed causal KV pages");
        if engine.cuda_graph_enabled() {
            let added = u64::from(position == 2046 || position == 2048);
            assert_eq!(
                engine.cuda_graph_stats()[2],
                before[2] + added,
                "graph cache did not distinguish its attention span"
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires MiniCPM5 checkpoint and a CUDA device"]
fn prefill_graph_replays_live_tokens_sequences_and_remapped_pages() -> Result<()> {
    let directory = std::path::PathBuf::from(std::env::var("LIA_TEST_MINICPM5_LONG_REFERENCE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(directory.join("info.json"))?)?;
    let tokens: Vec<u32> = serde_json::from_value(info["tokens"].clone())?;
    let positions: Vec<usize> = serde_json::from_value(info["positions"].clone())?;
    let bytes = std::fs::read(directory.join("logits.f32"))?;
    ensure!(bytes.len().is_multiple_of(4), "truncated reference");
    let reference: Vec<_> = bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().expect("four bytes")))
        .collect();
    let mut engine = Engine::load(&std::env::var("LIA_TEST_MINICPM5_MODEL")?, 1024)?;
    let vocabulary = engine.vocabulary_size();
    let index = positions
        .iter()
        .position(|&p| p == 255)
        .ok_or_else(|| anyhow::anyhow!("missing position 255 reference"))?;
    let first = reference[index * vocabulary..(index + 1) * vocabulary].to_vec();
    let second: Vec<_> = std::iter::once(tokens[0])
        .chain(tokens[1..256].iter().rev().copied())
        .collect();
    let mut processed = 0;
    while processed < 255 {
        let count = engine.prefill_chunk(255 - processed)?;
        engine.forward_many(&second[processed..processed + count], false)?;
        processed += count;
    }
    let expected = [first, engine.forward(second[255])?];
    engine.reset()?;
    engine.configure_paging(2, 512)?;
    let mut tables = [(0..16).collect::<Vec<_>>(), (16..32).collect::<Vec<_>>()];
    let before = engine.cuda_graph_stats();
    for start in (0..256).step_by(64) {
        let order = if start % 128 == 0 { [0, 1] } else { [1, 0] };
        let rows: Vec<_> = order
            .into_iter()
            .flat_map(|sequence| {
                let source = if sequence == 0 { &tokens } else { &second };
                (start..start + 64).map(move |position| PagedRow {
                    token: source[position],
                    position: position as u32,
                    sequence,
                })
            })
            .collect();
        let outputs: &[u32] = if start == 192 { &[63, 127] } else { &[] };
        let actual = engine.forward_paged(&rows, &tables, outputs)?;
        if start == 0 {
            // Move complete cached pages before replaying the same graph with
            // new token IDs, positions, sequence order and physical page IDs.
            for (source, destination) in [(0, 15), (1, 14), (16, 31), (17, 30)] {
                let payload = engine.read_kv_page(source)?;
                engine.write_kv_page(destination, &payload)?;
            }
            tables.iter_mut().for_each(|pages| pages.reverse());
        }
        if start == 192 {
            ensure!(
                actual.len() == 2 * vocabulary,
                "graph logit extent mismatch"
            );
            for (output, sequence) in actual.chunks_exact(vocabulary).zip(order) {
                let mut maximum = 0_f32;
                for (&a, &b) in output.iter().zip(&expected[sequence as usize]) {
                    ensure!(a.is_finite() && b.is_finite(), "nonfinite graph output");
                    maximum = maximum.max((a - b).abs());
                }
                println!("prefill graph sequence {sequence}: max_abs={maximum}");
                ensure!(
                    maximum < 0.025,
                    "prefill replay captured stale metadata or scratch"
                );
            }
        } else {
            ensure!(actual.is_empty(), "unexpected prefill logits");
        }
    }
    if engine.cuda_graph_enabled() && engine.buffers_preallocated() {
        assert_eq!(engine.cuda_graph_stats()[2], before[2] + 1);
        assert_eq!(engine.cuda_graph_stats()[0], before[0] + 3);
    }
    Ok(())
}
