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
    assert!(engine.forward_many(&[248320], false).is_err());
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
    if engine.cuda_graph_enabled() {
        let [launches, updates, instantiations] = engine.cuda_graph_stats();
        assert!(launches > 0 && updates > 0 && instantiations > 0);
        assert_eq!(launches, updates + instantiations);
    }
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
    if engine.cuda_graph_enabled() {
        let [launches, updates, instantiations] = engine.cuda_graph_stats();
        assert!(launches > 0 && updates > 0 && instantiations > 0);
        assert_eq!(launches, updates + instantiations);
    }
    Ok(())
}

#[test]
#[ignore = "requires LIA_TEST_MODEL and a CUDA device"]
fn stochastic_native_mtp_preserves_degenerate_target_and_rollback() -> Result<()> {
    use lia::sampling::Sampling;
    let path = std::env::var("LIA_TEST_MODEL")?;
    let mut engine = Engine::load(&path, 64)?;
    let mut expected = Vec::new();
    engine.forward(109266)?;
    for _ in 0..8 {
        let token = engine.current_token()?;
        expected.push(token);
        engine.forward(token)?;
    }
    engine.reset()?;
    engine.forward_many(&[109266], false)?;
    let mut target = Sampling::new(1.0, 1, 1.0, 42)?;
    let mut draft = Sampling::new(1.0, 1, 1.0, 42 ^ 0x85ebca6b)?;
    let mut first = engine.current_sample(&mut target)?;
    let mut actual = Vec::new();
    let (mut proposed, mut accepted) = (0, 0);
    while actual.len() < expected.len() {
        let (tokens, next, n, a) = engine.speculate_stochastic(
            first,
            expected.len() - actual.len(),
            248046,
            &mut target,
            &mut draft,
        )?;
        actual.extend(tokens);
        first = next;
        proposed += n;
        accepted += a;
    }
    engine.synchronize()?;
    assert_eq!(actual, expected);
    assert!(
        accepted > 0 && accepted < proposed,
        "must exercise acceptance and rollback"
    );
    if engine.cuda_graph_enabled() {
        let [launches, updates, instantiations] = engine.cuda_graph_stats();
        assert!(launches > 0 && updates > 0 && instantiations > 0);
        assert_eq!(launches, updates + instantiations);
    }
    Ok(())
}

#[test]
#[ignore = "requires LIA_TEST_MODEL and a CUDA device"]
fn prefill_workspace_and_reset_are_deterministic() -> Result<()> {
    let path = std::env::var("LIA_TEST_MODEL")?;
    let mut engine = Engine::load(&path, 1024)?;
    let configured = std::env::var("LIA_PREFILL_FP8").as_deref() == Ok("1");
    assert_eq!(engine.fp8_prefill(), configured);
    let prefix = vec![109266; 513];
    engine.forward_many(&prefix, false)?;
    let token = engine.current_token()?;
    let expected = engine.forward(token)?;
    assert!(expected.iter().all(|x| x.is_finite()));
    engine.reset()?;
    assert!(engine.forward_many(&[248320], false).is_err());
    engine.reset()?;
    engine.forward_many(&prefix, false)?;
    assert_eq!(engine.current_token()?, token);
    let actual = engine.forward(token)?;
    assert_eq!(
        actual, expected,
        "large prefill workspace/reset changes logits"
    );
    if engine.cuda_graph_enabled() {
        assert!(engine.cuda_graph_stats()[0] > 0);
    }
    println!(
        "large prefill and recovery verified; FP8={configured}, weight_workspace_bytes={}",
        engine.fp8_weight_workspace_bytes()
    );
    Ok(())
}
