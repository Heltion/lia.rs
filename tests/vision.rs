//! Real images and independent reference pixels remain outside the repository.
use anyhow::Result;
use lia::{gguf::Model, vision::Processor};

#[test]
#[ignore = "requires LIA_TEST_MM_PROJ and LIA_TEST_VISION_FIXTURES"]
fn real_image_preprocessing_matches_independent_reference() -> Result<()> {
    let model = Model::open(std::env::var("LIA_TEST_MM_PROJ")?)?;
    let processor = Processor::from_model(&model, 256, 1024)?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(std::env::var("LIA_TEST_VISION_FIXTURES")?)?)?;
    let images = manifest["images"].as_array().unwrap();
    assert!(!images.is_empty());
    for image in images {
        let input = std::fs::read(image["image"].as_str().unwrap())?;
        let actual = processor.prepare(&input)?;
        assert_eq!(actual.width as u64, image["width"].as_u64().unwrap());
        assert_eq!(actual.height as u64, image["height"].as_u64().unwrap());
        let expected = std::fs::read(image["pixels"].as_str().unwrap())?;
        assert_eq!(expected.len(), actual.pixels.len() * 4);
        for (index, (&value, bytes)) in actual
            .pixels
            .iter()
            .zip(expected.as_chunks::<4>().0)
            .enumerate()
        {
            let expected = u32::from_le_bytes(*bytes);
            assert_eq!(value.to_bits(), expected, "{} pixel {index}", image["id"]);
        }
        println!(
            "{}: {}x{}, {} normalized pixels match bitwise",
            image["id"],
            actual.width,
            actual.height,
            actual.pixels.len()
        );
    }
    Ok(())
}
