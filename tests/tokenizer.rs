use anyhow::{Context, Result};
use lia::tokenizer::Tokenizer;

#[test]
#[ignore = "requires LIA_TEST_GSQ_MODEL and an external official-tokenizer reference in LIA_TOKENIZER_REFERENCE"]
fn huggingface_tokenizer_matches_official_reference() -> Result<()> {
    let tokenizer = Tokenizer::from_huggingface(std::env::var("LIA_TEST_GSQ_MODEL")?)?;
    let reference: serde_json::Value = serde_json::from_reader(std::fs::File::open(
        std::env::var("LIA_TOKENIZER_REFERENCE")?,
    )?)?;
    for case in reference["cases"]
        .as_array()
        .context("missing tokenizer reference cases")?
    {
        let text = case["text"].as_str().context("missing reference text")?;
        let special = case["parse_special"]
            .as_bool()
            .context("missing reference token mode")?;
        let expected = case["ids"]
            .as_array()
            .context("missing reference IDs")?
            .iter()
            .map(|v| Ok(u32::try_from(v.as_u64().context("invalid reference ID")?)?))
            .collect::<Result<Vec<_>>>()?;
        let ids = tokenizer.encode(text, special)?;
        assert_eq!(
            ids, expected,
            "tokenizer mismatch for {text:?}, parse_special={special}"
        );
        assert_eq!(
            tokenizer.encode(text, special)?,
            expected,
            "cached tokenization differs"
        );
        let decoded = String::from_utf8(tokenizer.decode(&ids)?)?;
        assert_eq!(
            decoded,
            case["decoded"]
                .as_str()
                .context("missing reference decode")?
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires external checkpoint and official chat-template reference"]
fn model_chat_template_matches_official_reference() -> Result<()> {
    let model = std::path::PathBuf::from(std::env::var("LIA_TEST_GSQ_MODEL")?);
    let template = std::fs::read_to_string(model.join("chat_template.jinja"))?;
    let reference: serde_json::Value = serde_json::from_reader(std::fs::File::open(
        std::env::var("LIA_CHAT_TEMPLATE_REFERENCE")?,
    )?)?;
    let bos = reference["bos_token"]
        .as_str()
        .context("missing BOS token")?;
    let tokenizer = Tokenizer::from_huggingface(&model)?;
    for case in reference["cases"]
        .as_array()
        .context("missing template cases")?
    {
        let prompt = lia::chat::render_template_with_bos(&template, &case["request"], bos)?;
        assert_eq!(
            prompt,
            case["prompt"]
                .as_str()
                .context("missing reference prompt")?
        );
        let tokens = tokenizer.encode(&prompt, true)?;
        let expected: Vec<u32> = serde_json::from_value(case["ids"].clone())?;
        assert_eq!(tokens, expected);
    }
    Ok(())
}
