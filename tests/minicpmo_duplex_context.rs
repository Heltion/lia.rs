use anyhow::{Context, Result, ensure};
use lia::models::minicpm_o45::duplex::context::{reference_features, system_texts};

#[test]
#[ignore = "requires official external duplex system context preprocessing oracle"]
fn reference_system_context_matches_official_processor() -> Result<()> {
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_DUPLEX_CONTEXT_ORACLE")?);
    let info: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(root.join("info.json")).context("opening context oracle")?,
    )?;
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let bytes = std::fs::read(std::path::Path::new(&model).join("assets/HT_ref_audio.wav"))?;
    let features = reference_features(&bytes)?;
    ensure!(
        features.frames == info["frames"].as_u64().context("missing frames")? as usize,
        "reference context frame count differs"
    );
    let mut maxima = Vec::new();
    for (file, tolerance) in [("numpy-mel.f32", 1e-5f32), ("mel.f32", 5e-4)] {
        let expected = std::fs::read(root.join(file))?;
        ensure!(
            expected.len() == features.values.len() * 4,
            "reference mel extent differs"
        );
        let maximum = features
            .values
            .iter()
            .zip(expected.chunks_exact(4))
            .map(|(a, b)| (a - f32::from_le_bytes(b.try_into().unwrap())).abs())
            .fold(0.0f32, f32::max);
        ensure!(
            maximum < tolerance,
            "duplex context {file} mel error {maximum}"
        );
        maxima.push((file, maximum));
    }
    let tokenizer = lia::tokenizer::Tokenizer::from_huggingface(&model)?;
    for case in info["templates"].as_array().context("missing templates")? {
        let (prefix, suffix) = system_texts(
            case["instructions"]
                .as_str()
                .context("missing instructions")?,
        );
        for (text, key) in [(&prefix, "prefix_ids"), (&suffix, "suffix_ids")] {
            let expected: Vec<u32> = serde_json::from_value(case[key].clone())?;
            ensure!(
                tokenizer.encode(text, true)? == expected,
                "duplex {key} differs from official template"
            );
        }
    }
    println!(
        "duplex reference system context: frames={} rows={} mel_errors={maxima:?}",
        features.frames,
        features.frames.div_ceil(2) / 5
    );
    Ok(())
}
