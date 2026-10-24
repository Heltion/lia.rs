//! External model/image validation; fixtures and exported embeddings stay outside Git.
#![cfg(feature = "cuda")]
use anyhow::{Result, ensure};
use lia::{gguf::Model, inference::Engine, vision::Processor};

#[test]
#[ignore = "requires language/vision models, fixture manifest and external output directory"]
fn real_projector_encodes_selected_images() -> Result<()> {
    let projector = std::env::var("LIA_TEST_MM_PROJ")?;
    let language = std::env::var("LIA_TEST_MODEL")?;
    let output = std::path::PathBuf::from(std::env::var("LIA_TEST_VISION_OUTPUT")?);
    std::fs::create_dir_all(&output)?;
    ensure!(
        !output
            .canonicalize()?
            .starts_with(std::path::Path::new(env!("CARGO_MANIFEST_DIR"))),
        "vision validation outputs must remain outside the repository"
    );
    let model = Model::open(&projector)?;
    let processor = Processor::from_model(&model, 256, 1024)?;
    let mut engine = Engine::load_with_vision(&language, 32768, &projector)?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(std::env::var("LIA_TEST_VISION_FIXTURES")?)?)?;
    for image in manifest["images"].as_array().unwrap() {
        let id = image["id"].as_str().unwrap();
        ensure!(
            !id.contains('/') && !id.contains('\\'),
            "invalid fixture id"
        );
        let prepared = processor.prepare(&std::fs::read(image["image"].as_str().unwrap())?)?;
        let embeddings = engine.image_embeddings(&prepared)?;
        assert_eq!(
            embeddings.len(),
            prepared.width * prepared.height / 1024 * 5120
        );
        assert!(embeddings.iter().all(|x| x.is_finite()));
        assert!(embeddings.iter().any(|x| x.abs() > 0.01));
        let path = output.join(format!("{id}.f32"));
        ensure!(
            !path.exists(),
            "refusing to overwrite earlier validation output"
        );
        let bytes: Vec<u8> = embeddings.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(path, bytes)?;
        println!(
            "{id}: {}x{}, {} projected features",
            prepared.width,
            prepared.height,
            embeddings.len()
        );
    }
    Ok(())
}
