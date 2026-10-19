use super::*;
use crate::sampling::Sampling;

pub(super) fn presence(request: &Json) -> Result<f32> {
    let default = match std::env::var("LIA_DEFAULT_PRESENCE_PENALTY") {
        Ok(v) => v
            .parse::<f64>()
            .map_err(|e| anyhow!("invalid LIA_DEFAULT_PRESENCE_PENALTY: {e}"))?,
        Err(std::env::VarError::NotPresent) => 0.0,
        Err(e) => return Err(anyhow!("invalid LIA_DEFAULT_PRESENCE_PENALTY: {e}")),
    };
    presence_with_default(request, default)
}

fn presence_with_default(request: &Json, default: f64) -> Result<f32> {
    let value = request
        .get("presence_penalty")
        .filter(|v| !v.is_null())
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| anyhow!("invalid presence_penalty"))
        })
        .transpose()?
        .unwrap_or(default);
    anyhow::ensure!(
        value.is_finite() && (-2.0..=2.0).contains(&value),
        "invalid presence_penalty"
    );
    Ok(value as f32)
}

pub(super) fn from_request(request: &Json) -> Result<Option<Sampling>> {
    let default_temperature = match std::env::var("LIA_DEFAULT_TEMPERATURE") {
        Ok(value) => value
            .parse::<f64>()
            .map_err(|error| anyhow!("invalid LIA_DEFAULT_TEMPERATURE: {error}"))?,
        Err(std::env::VarError::NotPresent) => 1.0,
        Err(error) => return Err(anyhow!("invalid LIA_DEFAULT_TEMPERATURE: {error}")),
    };
    with_temperature_default(request, default_temperature)
}

fn with_temperature_default(request: &Json, default_temperature: f64) -> Result<Option<Sampling>> {
    let number = |name: &str, default: f64| -> Result<f64> {
        request
            .get(name)
            .filter(|v| !v.is_null())
            .map(|v| v.as_f64().ok_or_else(|| anyhow!("invalid {name}")))
            .transpose()
            .map(|v| v.unwrap_or(default))
    };
    let temperature = number("temperature", default_temperature)?;
    presence(request)?;
    anyhow::ensure!(
        temperature.is_finite() && temperature >= 0.0,
        "invalid temperature"
    );
    for (name, default) in [
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
    fn deployment_presence_preserves_explicit_overrides() -> Result<()> {
        assert_eq!(presence_with_default(&json!({}), 0.5)?, 0.5);
        assert_eq!(
            presence_with_default(&json!({"presence_penalty":0}), 0.5)?,
            0.0
        );
        assert!(presence_with_default(&json!({}), f64::NAN).is_err());
        assert!(presence_with_default(&json!({}), 2.1).is_err());
        Ok(())
    }
    #[test]
    fn deployment_temperature_preserves_request_overrides() -> Result<()> {
        assert_eq!(
            with_temperature_default(&json!({}), 0.6)?
                .unwrap()
                .settings(),
            (0.6, 20, 0.95)
        );
        assert_eq!(
            with_temperature_default(&json!({"temperature": 1.0}), 0.6)?
                .unwrap()
                .settings(),
            (1.0, 20, 0.95)
        );
        assert!(with_temperature_default(&json!({"temperature": 0.0}), 0.6)?.is_none());
        for invalid in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(with_temperature_default(&json!({}), invalid).is_err());
        }
        Ok(())
    }

    #[test]
    fn sampling_request_validation() -> Result<()> {
        assert_eq!(
            from_request(&json!({}))?.unwrap().settings(),
            (1.0, 20, 0.95)
        );
        assert!(from_request(&json!({"temperature":0.0}))?.is_none());
        let s =
            from_request(&json!({"temperature":1.0,"top_k":20,"top_p":0.95,"seed":42}))?.unwrap();
        assert_eq!(s.settings(), (1.0, 20, 0.95));
        for invalid in [
            json!({"temperature":-1.0}),
            json!({"temperature":"1"}),
            json!({"temperature":1.0,"top_k":21}),
            json!({"temperature":1.0,"top_p":0.0}),
            json!({"temperature":1.0,"seed":-1}),
            json!({"temperature":1.0,"presence_penalty":2.1}),
        ] {
            assert!(from_request(&invalid).is_err());
        }
        Ok(())
    }
}
