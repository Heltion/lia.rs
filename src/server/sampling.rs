use super::*;
use crate::sampling::Sampling;

pub(super) fn from_request(request: &Json) -> Result<Option<Sampling>> {
    let number = |name: &str, default: f64| -> Result<f64> {
        request
            .get(name)
            .filter(|v| !v.is_null())
            .map(|v| v.as_f64().ok_or_else(|| anyhow!("invalid {name}")))
            .transpose()
            .map(|v| v.unwrap_or(default))
    };
    let temperature = number("temperature", 0.0)?;
    anyhow::ensure!(
        temperature.is_finite() && temperature >= 0.0,
        "invalid temperature"
    );
    for (name, default) in [
        ("presence_penalty", 0.0),
        ("frequency_penalty", 0.0),
        ("repetition_penalty", 1.0),
        ("min_p", 0.0),
    ] {
        anyhow::ensure!(number(name, default)? == default, "unsupported {name}");
    }
    if temperature == 0.0 {
        return Ok(None);
    }
    let integer = |name: &str, default: u64| -> Result<u64> {
        request
            .get(name)
            .filter(|v| !v.is_null())
            .map(|v| v.as_u64().ok_or_else(|| anyhow!("invalid {name}")))
            .transpose()
            .map(|v| v.unwrap_or(default))
    };
    let top_k = usize::try_from(integer("top_k", 20)?)?;
    Ok(Some(Sampling::new(
        temperature,
        top_k,
        number("top_p", 0.95)?,
        integer("seed", 42)?,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_request_validation() -> Result<()> {
        assert!(from_request(&json!({}))?.is_none());
        let s =
            from_request(&json!({"temperature":1.0,"top_k":20,"top_p":0.95,"seed":42}))?.unwrap();
        assert_eq!(s.settings(), (1.0, 20, 0.95));
        for invalid in [
            json!({"temperature":-1.0}),
            json!({"temperature":"1"}),
            json!({"temperature":1.0,"top_k":21}),
            json!({"temperature":1.0,"top_p":0.0}),
            json!({"temperature":1.0,"seed":-1}),
            json!({"temperature":1.0,"presence_penalty":1.0}),
        ] {
            assert!(from_request(&invalid).is_err());
        }
        Ok(())
    }
}
