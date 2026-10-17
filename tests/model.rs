#![cfg(feature = "cuda")]
use anyhow::Result;
use lia::inference::Engine;

#[test]
#[ignore = "requires LIA_TEST_MODEL and a CUDA device"]
fn batched_forward_and_native_mtp_match_greedy() -> Result<()> {
    let path = std::env::var("LIA_TEST_MODEL")?;
    let initial = 109266;
    let mut engine = Engine::load(&path, 64)?;
    let mut expected = Vec::new();
    engine.forward(initial)?;
    for _ in 0..8 {
        let next = engine.current_token()?;
        expected.push(next);
        engine.forward(next)?;
    }
    engine.reset()?;
    let mut prefix = vec![initial];
    prefix.extend_from_slice(&expected[..expected.len() - 1]);
    let logits = engine.forward_many(&prefix, true)?;
    assert_eq!(logits.len(), prefix.len() * 248320);
    for (row, expected) in logits.as_chunks::<248320>().0.iter().zip(&expected) {
        let actual = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        assert_eq!(actual, *expected, "batched target differs from greedy");
    }
    engine.reset()?;
    engine.forward(initial)?;
    let mut first = engine.current_token()?;
    let mut actual = Vec::new();
    let (mut proposed, mut accepted) = (0, 0);
    while actual.len() < expected.len() {
        let (tokens, next, n, a) =
            engine.speculate(first, expected.len() - actual.len(), 248046)?;
        actual.extend(tokens);
        first = next;
        proposed += n;
        accepted += a;
    }
    engine.synchronize()?;
    assert_eq!(actual, expected, "native MTP differs from greedy");
    assert!(proposed > 0 && accepted > 0 && accepted < proposed);
    Ok(())
}

#[test]
#[ignore = "requires LIA_TEST_MODEL and a CUDA device"]
fn sampled_native_mtp_matches_single_token_sampling() -> Result<()> {
    use lia::sampling::Sampling;
    let path = std::env::var("LIA_TEST_MODEL")?;
    let mut engine = Engine::load(&path, 64)?;
    let mut reference = Sampling::new(1.0, 20, 0.95, 42)?;
    let mut logits = engine.forward(109266)?;
    let mut expected = Vec::new();
    for step in 0..8 {
        let distribution = reference.distribution(
            logits
                .iter()
                .copied()
                .enumerate()
                .map(|(id, value)| (id as u32, value)),
        )?;
        let token = reference.draw(&distribution);
        expected.push(token);
        if token == 248046 {
            break;
        }
        if step < 7 {
            logits = engine.forward(token)?;
        }
    }
    engine.reset()?;
    engine.forward_many(&[109266], false)?;
    let mut sampling = Sampling::new(1.0, 20, 0.95, 42)?;
    let mut first = engine.current_sample(&mut sampling)?;
    let mut actual = Vec::new();
    let (mut proposed, mut accepted) = (0, 0);
    while actual.len() < expected.len() {
        if first == 248046 {
            actual.push(first);
            break;
        }
        let (tokens, next, n, a) = engine.speculate_sampled(
            first,
            expected.len() - actual.len(),
            248046,
            &mut sampling,
        )?;
        actual.extend(tokens);
        first = next;
        proposed += n;
        accepted += a;
        if actual.last() == Some(&248046) {
            break;
        }
    }
    engine.synchronize()?;
    assert_eq!(
        actual, expected,
        "sample-and-match changed the target random stream"
    );
    assert!(proposed > 0 && accepted <= proposed);
    Ok(())
}
