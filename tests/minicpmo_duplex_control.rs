use anyhow::{Context, Result, ensure};
use lia::models::minicpm_o45::duplex::{Control, UNIT_END};
use serde_json::Value;
use std::collections::VecDeque;
#[test]
#[ignore = "requires external official MiniCPMODuplex control oracle"]
fn control_matches_official_streaming_generate() -> Result<()> {
    let path = std::env::var("LIA_MINICPMO_DUPLEX_CONTROL_ORACLE")?;
    let info: Value = serde_json::from_reader(std::fs::File::open(path)?)?;
    ensure!(
        info["unit_end"].as_u64() == Some(u64::from(UNIT_END)),
        "unit-end token mismatch"
    );
    for case in info["cases"].as_array().context("missing control cases")? {
        let mut control = Control::new(
            case["limit"].as_u64().context("missing limit")? as usize,
            case["force_listen_count"]
                .as_u64()
                .context("missing force-listen count")? as usize,
            case["initial_turn_ended"]
                .as_bool()
                .context("missing initial turn state")?,
        )?;
        for unit in case["units"].as_array().context("missing units")? {
            let mut samples: VecDeque<u32> = serde_json::from_value(unit["sampled"].clone())?;
            let (mut feeds, mut hidden, mut drawn) = (Vec::new(), Vec::new(), Vec::new());
            control.begin()?;
            let last = loop {
                let sample = if control.needs_sample() {
                    let token = samples
                        .pop_front()
                        .context("control exhausted fixture draws")?;
                    drawn.push(token);
                    Some(token)
                } else {
                    None
                };
                let decision = control.step(sample)?;
                feeds.push(decision.token);
                if decision.collect_hidden {
                    hidden.push(decision.token);
                }
                if decision.stop {
                    break decision;
                }
            };
            feeds.push(UNIT_END);
            ensure!(
                serde_json::to_value(&feeds)? == unit["feeds"]
                    && serde_json::to_value(&hidden)? == unit["hidden_tokens"]
                    && serde_json::to_value(&hidden)? == unit["generated_tokens"]
                    && serde_json::to_value(&drawn)? == unit["drawn"]
                    && unit["is_listen"] == last.is_listen
                    && unit["end_of_turn"] == last.end_of_turn
                    && unit["turn_ended"] == control.turn_ended(),
                "control mismatch in {}",
                case["name"]
            );
            println!(
                "{}: feeds={feeds:?}, hidden={hidden:?}, listen={}, end_of_turn={}",
                case["name"], last.is_listen, last.end_of_turn
            );
        }
    }
    Ok(())
}
