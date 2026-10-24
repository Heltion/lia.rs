use anyhow::{Context, Result, ensure};
use lia::models::minicpm_o45::duplex::drain::{Drain, TextTail, budget};
use serde_json::Value;
#[test]
#[ignore = "requires external pinned vLLM-Omni duplex stopping oracle"]
fn stopping_budgets_and_sentence_tails_match_reference() -> Result<()> {
    let info: Value = serde_json::from_reader(std::fs::File::open(std::env::var(
        "LIA_MINICPMO_DUPLEX_DRAIN_ORACLE",
    )?)?)?;
    let cases = info["cases"].as_array().context("missing drain cases")?;
    for case in cases {
        let drain = Drain::for_turn(
            case["native"] == true,
            case["end"] == true,
            case["sentence"] == true,
        );
        let history: Vec<u32> = serde_json::from_value(case["history"].clone())?;
        let actual = drain.decision(
            case["step"].as_u64().context("missing drain step")? as usize,
            &history,
        );
        ensure!(
            case["kind"].as_u64() == Some(drain as u64)
                && case["silent_run"].as_u64() == Some(actual.silent_run as u64)
                && case["quiet_run"].as_u64() == Some(actual.quiet_run as u64)
                && case["stop"] == actual.stop
                && case["mask"] == actual.mask_eos,
            "drain decision mismatch: {case}"
        );
    }
    for case in info["budgets"].as_array().context("missing budget cases")? {
        let (maximum, minimum) = budget(case["start"] == true, case["end"] == true);
        ensure!(
            case["maximum"].as_u64() == Some(maximum as u64)
                && case["minimum"].as_u64() == Some(minimum as u64),
            "duplex budget mismatch"
        );
    }
    for parts in info["texts"].as_array().context("missing text cases")? {
        let mut tail = TextTail::default();
        for part in parts.as_array().context("invalid text sequence")? {
            tail.push(part["text"].as_str().context("missing appended text")?);
            ensure!(
                part["tail"] == tail.text() && part["ends"] == tail.ends_sentence(),
                "sentence-tail mismatch: {part}"
            );
        }
        tail.reset();
        ensure!(
            tail.text().is_empty() && !tail.ends_sentence(),
            "sentence tail reset failed"
        );
    }
    println!(
        "{} drain decisions and reference sentence/budget cases passed",
        cases.len()
    );
    Ok(())
}
