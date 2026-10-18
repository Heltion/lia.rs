#![cfg(feature = "cuda")]
use anyhow::Result;
use lia::inference::Engine;

#[test]
#[ignore = "requires LIA_TEST_MODEL and a CUDA device with long-context headroom"]
fn native_mtp_and_snapshot_replay_reach_the_48k_boundary() -> Result<()> {
    let path = std::env::var("LIA_TEST_MODEL")?;
    let mut engine = Engine::load(&path, 49152)?;
    assert_eq!(engine.capacity(), 49152);
    let mut remaining = 49144;
    while remaining > 0 {
        let batch = engine.prefill_chunk(remaining)?;
        engine.forward_many(&vec![109266; batch], false)?;
        remaining -= batch;
        if remaining % 8192 < batch {
            eprintln!("long-context prefill: {} tokens", 49144 - remaining);
        }
    }
    let saved = engine.snapshot()?;
    let first = engine.current_token()?;
    let mut next = first;
    let mut actual = Vec::new();
    let (mut proposed, mut accepted) = (0, 0);
    while actual.len() < 8 {
        // No early stop: all eight positions must reach the final cache slot.
        let (tokens, following, n, a) = engine.speculate(next, 8 - actual.len(), u32::MAX)?;
        assert!(!tokens.is_empty());
        actual.extend(tokens);
        next = following;
        proposed += n;
        accepted += a;
    }
    assert_eq!(actual.len(), 8);
    assert!(proposed > 0 && accepted <= proposed);
    engine.restore(&saved)?;
    let mut next = first;
    for token in actual {
        assert_eq!(token, next, "long-context MTP changed greedy output");
        let logits = engine.forward(token)?;
        assert!(logits.iter().all(|value| value.is_finite()));
        next = engine.current_token()?;
    }
    assert!(engine.forward(next).is_err());
    assert!(engine.forward_many(&[next], false).is_err());
    engine.restore(&saved)?;
    assert_eq!(engine.current_token()?, first);
    engine.reset()?;
    assert!(engine.forward(109266)?.iter().all(|v| v.is_finite()));
    engine.synchronize()?;
    Ok(())
}
