use anyhow::{Context, Result, ensure};
use lia::sampling::Sampling;

#[test]
#[ignore = "requires LIA_TEST_SAMPLER_REFERENCE with external GPU sampler fixtures"]
fn position_sampler_matches_reference_choices() -> Result<()> {
    let path =
        std::env::var("LIA_TEST_SAMPLER_REFERENCE").context("set LIA_TEST_SAMPLER_REFERENCE")?;
    let data: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).with_context(|| format!("reading {path}"))?)
            .context("parsing sampler reference")?;
    let cases = data["cases"]
        .as_array()
        .context("missing reference cases")?;
    ensure!(!cases.is_empty(), "empty sampler reference");
    for (index, row) in cases.iter().enumerate() {
        let mut sampler = Sampling::new(
            row["temperature"].as_f64().context("invalid temperature")?,
            usize::try_from(row["top_k"].as_u64().context("invalid top_k")?)?,
            row["top_p"].as_f64().context("invalid top_p")?,
            row["seed"].as_u64().context("invalid seed")?,
        )?
        .with_position_gumbel();
        sampler.set_position(row["position"].as_u64().context("invalid position")?);
        let candidates = row["candidates"]
            .as_array()
            .context("missing candidates")?
            .iter()
            .map(|pair| {
                Ok((
                    u32::try_from(pair[0].as_u64().context("invalid token")?)?,
                    pair[1].as_f64().context("invalid logit")? as f32,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let distribution = sampler.distribution(candidates)?;
        let expected = row["expected"]
            .as_u64()
            .context("missing reference choice")?;
        ensure!(
            u64::from(sampler.draw(&distribution)) == expected,
            "sampler choice differs for case {index}"
        );
        let expected_next = row["expected_next"]
            .as_u64()
            .context("missing next reference choice")?;
        ensure!(
            u64::from(sampler.draw(&distribution)) == expected_next,
            "position advancement differs for case {index}"
        );
        sampler.set_position(row["position"].as_u64().context("invalid position")?);
        ensure!(
            u64::from(sampler.draw(&distribution)) == expected,
            "position reset differs for case {index}"
        );
    }
    eprintln!(
        "Position sampler matched all {} GPU reference choices",
        cases.len()
    );
    Ok(())
}
