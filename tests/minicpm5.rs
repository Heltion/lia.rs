#![cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};
use lia::inference::Engine;
use lia::inference::PagedRow;

#[test]
#[ignore = "requires MiniCPM5 checkpoint and external independent FP32 logits"]
fn minicpm5_logits_match_reference_and_prefill() -> Result<()> {
    let model = std::env::var("LIA_TEST_MINICPM5_MODEL")?;
    let directory = std::path::PathBuf::from(std::env::var("LIA_TEST_MINICPM5_REFERENCE")?);
    let info: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(directory.join("info.json")).context("opening reference metadata")?,
    )?;
    let tokens = info["tokens"]
        .as_array()
        .context("missing reference tokens")?
        .iter()
        .map(|v| Ok(u32::try_from(v.as_u64().context("invalid token")?)?))
        .collect::<Result<Vec<_>>>()?;
    let bytes = std::fs::read(directory.join("logits.f32"))?;
    ensure!(bytes.len() % 4 == 0, "truncated reference logits");
    let reference = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")))
        .collect::<Vec<_>>();
    let mut engine = Engine::load(&model, 64)?;
    let vocabulary = engine.vocabulary_size();
    ensure!(
        reference.len() == tokens.len() * vocabulary,
        "reference shape mismatch"
    );
    let mut sequential = Vec::new();
    for (position, &token) in tokens.iter().enumerate() {
        let actual = engine.forward(token)?;
        let expected = &reference[position * vocabulary..(position + 1) * vocabulary];
        compare(
            &actual,
            expected,
            0.025,
            &format!("reference position {position}"),
        )?;
        sequential.extend(actual);
    }
    engine.reset()?;
    let batched = engine.forward_many(&tokens, true)?;
    compare(&batched, &sequential, 0.025, "prefill versus sequential")?;
    engine.synchronize()?;
    Ok(())
}

fn compare(actual: &[f32], expected: &[f32], tolerance: f32, label: &str) -> Result<()> {
    ensure!(actual.len() == expected.len(), "{label}: shape mismatch");
    let mut maximum = 0.0_f32;
    let mut squared = 0.0_f64;
    for (&a, &b) in actual.iter().zip(expected) {
        ensure!(a.is_finite() && b.is_finite(), "{label}: nonfinite logit");
        maximum = maximum.max((a - b).abs());
        squared += f64::from(a - b).powi(2);
    }
    let rms = (squared / actual.len() as f64).sqrt();
    println!("{label}: max_abs={maximum:.6} rms={rms:.6}");
    ensure!(maximum < tolerance, "{label}: logits exceed tolerance");
    Ok(())
}

#[test]
#[ignore = "requires MiniCPM5 checkpoint and a CUDA device"]
fn paged_requests_match_independent_sequences() -> Result<()> {
    let model = std::env::var("LIA_TEST_MINICPM5_MODEL")?;
    let sequences = [vec![0, 36417, 1782, 22], vec![0, 75828, 68572, 1398, 35]];
    let mut engine = Engine::load(&model, 128)?;
    let mut expected = Vec::new();
    for sequence in &sequences {
        engine.reset()?;
        for &token in &sequence[..sequence.len() - 1] {
            engine.forward(token)?;
        }
        expected.extend(engine.forward(*sequence.last().expect("nonempty"))?);
    }
    engine.reset()?;
    engine.configure_paging(2, 64)?;
    // Reverse physical page order; contiguous layout must never be assumed.
    let tables = [vec![3, 1], vec![2, 0]];
    let mut rows = Vec::new();
    let mut outputs = Vec::new();
    for (sequence, tokens) in sequences.iter().enumerate() {
        for (position, &token) in tokens.iter().enumerate() {
            rows.push(PagedRow {
                token,
                position: position as u32,
                sequence: sequence as u32,
            });
        }
        outputs.push((rows.len() - 1) as u32);
    }
    let actual = engine.forward_paged(&rows, &tables, &outputs)?;
    compare(&actual, &expected, 0.025, "isolated paged prefill")?;
    let saved = engine.read_kv_page(3)?;
    engine.write_kv_pages(&[(1, &saved), (0, &saved)])?;
    assert!(engine.write_kv_pages(&[(1, &saved), (1, &saved)]).is_err());
    assert_eq!(
        engine.read_kv_page(0)?,
        saved,
        "window KV upload changed bits"
    );
    assert_eq!(
        engine.read_kv_page(1)?,
        saved,
        "KV page transfer changed bits"
    );
    // Interleaved decoding at different logical positions.
    let follow = [22, 35];
    let rows: Vec<_> = (0..2)
        .map(|sequence| PagedRow {
            token: follow[sequence],
            position: sequences[sequence].len() as u32,
            sequence: sequence as u32,
        })
        .collect();
    let actual = engine.forward_paged(&rows, &tables, &[0, 1])?;
    let a = engine.read_kv_page(3)?;
    let b = engine.read_kv_page(2)?;
    engine.write_kv_page(1, &a)?;
    engine.write_kv_page(0, &b)?;
    let before = engine.cuda_graph_stats();
    let again = [35, 22];
    let rows: Vec<_> = (0..2)
        .map(|sequence| PagedRow {
            token: again[sequence],
            position: sequences[sequence].len() as u32 + 1,
            sequence: sequence as u32,
        })
        .collect();
    let replayed = engine.forward_paged(&rows, &[vec![1, 3], vec![0, 2]], &[0, 1])?;
    if engine.cuda_graph_enabled() {
        assert_eq!(
            engine.cuda_graph_stats()[0],
            before[0] + 1,
            "decode did not replay its graph"
        );
        assert_eq!(
            engine.cuda_graph_stats()[2],
            before[2],
            "metadata change unnecessarily recaptured decode"
        );
    }
    drop(engine);
    let mut engine = Engine::load(&model, 128)?;
    let mut expected = Vec::new();
    for (sequence, tokens) in sequences.iter().enumerate() {
        engine.reset()?;
        engine.forward_many(tokens, false)?;
        expected.extend(engine.forward(follow[sequence])?);
    }
    compare(&actual, &expected, 0.025, "isolated paged decode")?;
    let mut expected = Vec::new();
    for (sequence, tokens) in sequences.iter().enumerate() {
        engine.reset()?;
        engine.forward_many(tokens, false)?;
        engine.forward(follow[sequence])?;
        expected.extend(engine.forward(again[sequence])?);
    }
    compare(
        &replayed,
        &expected,
        0.025,
        "decode replay after page relocation",
    )?;
    Ok(())
}

#[test]
#[ignore = "requires external long-context FP32 oracle and MiniCPM5 checkpoint"]
fn page_and_attention_partition_boundaries_match_reference() -> Result<()> {
    let directory = std::path::PathBuf::from(std::env::var("LIA_TEST_MINICPM5_LONG_REFERENCE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(directory.join("info.json"))?)?;
    let tokens: Vec<u32> = serde_json::from_value(info["tokens"].clone())?;
    let positions: Vec<usize> = serde_json::from_value(info["positions"].clone())?;
    let bytes = std::fs::read(directory.join("logits.f32"))?;
    ensure!(bytes.len().is_multiple_of(4), "truncated reference");
    let expected: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")))
        .collect();
    let mut engine = Engine::load(
        &std::env::var("LIA_TEST_MINICPM5_MODEL")?,
        (tokens.len() + 32).next_power_of_two(),
    )?;
    let vocabulary = engine.vocabulary_size();
    ensure!(
        expected.len() == positions.len() * vocabulary,
        "reference extent mismatch"
    );
    for (position, &token) in tokens.iter().enumerate() {
        let logits = engine.forward(token)?;
        if let Some(index) = positions.iter().position(|&p| p == position) {
            compare(
                &logits,
                &expected[index * vocabulary..(index + 1) * vocabulary],
                0.025,
                &format!("boundary {position}"),
            )?;
        }
    }
    engine.reset()?;
    let mut done = 0;
    while done < tokens.len() - 1 {
        let chunk = engine.prefill_chunk(tokens.len() - 1 - done)?;
        engine.forward_many(&tokens[done..done + chunk], false)?;
        done += chunk;
    }
    let logits = engine.forward(*tokens.last().context("empty reference")?)?;
    let index = positions
        .iter()
        .position(|&p| p == tokens.len() - 1)
        .context("missing final oracle row")?;
    compare(
        &logits,
        &expected[index * vocabulary..(index + 1) * vocabulary],
        0.025,
        "partitioned prefill",
    )?;
    engine.reset()?;
    let capacity = (tokens.len() + 32).next_power_of_two();
    engine.configure_paging(8, capacity)?;
    let pages: Vec<_> = (0..tokens.len().div_ceil(32) as u32).rev().collect();
    let rows: Vec<_> = tokens
        .iter()
        .enumerate()
        .map(|(position, &token)| PagedRow {
            token,
            position: position as u32,
            sequence: 0,
        })
        .collect();
    let outputs: Vec<_> = positions.iter().map(|&p| p as u32).collect();
    let actual = engine.forward_paged(&rows, &[pages], &outputs)?;
    compare(
        &actual,
        &expected,
        0.025,
        "paged fused prefill oracle positions",
    )?;
    Ok(())
}

#[test]
#[ignore = "requires external MiniCPM5 reference and CUDA device"]
fn small_batch_variants_match_the_same_independent_reference() -> Result<()> {
    let directory = std::path::PathBuf::from(std::env::var("LIA_TEST_MINICPM5_BATCH_REFERENCE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(directory.join("info.json"))?)?;
    let tokens: Vec<u32> = serde_json::from_value(info["tokens"].clone())?;
    let bytes = std::fs::read(directory.join("logits.f32"))?;
    let mut engine = Engine::load(&std::env::var("LIA_TEST_MINICPM5_MODEL")?, 256)?;
    let vocabulary = engine.vocabulary_size();
    ensure!(
        tokens.len() == 8 && bytes.len() == tokens.len() * vocabulary * 4,
        "invalid batch reference extent"
    );
    let expected: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")))
        .collect();
    engine.configure_paging(8, 32)?;
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
        let actual = engine.forward_paged(&rows, &tables, &outputs)?;
        for (sequence, row) in actual.chunks_exact(vocabulary).enumerate() {
            compare(
                row,
                &expected[sequence * vocabulary..(sequence + 1) * vocabulary],
                0.025,
                &format!("batch {batch} sequence {sequence}"),
            )?;
        }
    }
    Ok(())
}
