use anyhow::{Result, ensure};
use lia::models::minicpm_o45::audio::{Normalization, Processor};

#[test]
#[ignore = "requires official external MiniCPM-o audio preprocessing fixtures"]
fn official_numpy_features() -> Result<()> {
    let root = std::path::PathBuf::from(std::env::var("LIA_AUDIO_MEL_ORACLE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let processor = Processor::default();
    for case in info["cases"].as_array().unwrap() {
        let bytes = std::fs::read(root.join(case["input"].as_str().unwrap()))?;
        let samples: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let mode = if case["dynamic"] == true {
            Normalization::Dynamic
        } else {
            Normalization::Fixed
        };
        let actual = processor.extract(&samples, mode)?;
        let bytes = std::fs::read(root.join(case["mel"].as_str().unwrap()))?;
        ensure!(
            actual.frames == case["shape"][2].as_u64().unwrap() as usize
                && bytes.len() == actual.values.len() * 4,
            "wrong mel dimensions"
        );
        let mut max = 0.0f32;
        for (a, b) in actual.values.iter().zip(bytes.chunks_exact(4)) {
            max = max.max((a - f32::from_le_bytes(b.try_into().unwrap())).abs());
        }
        println!("{}: max_abs={max}", case["name"]);
        ensure!(max < 1e-5, "mel reference mismatch");
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires MiniCPM-o AWQ checkpoint, official Whisper oracle and CUDA"]
fn native_whisper_projector_matches_official_reference() -> Result<()> {
    let model = std::env::var("LIA_MINICPMO45_MODEL")?;
    let root = std::path::PathBuf::from(std::env::var("LIA_MINICPMO_WHISPER_ORACLE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let processor = Processor::default();
    let mut engine = lia::inference::Engine::load_huggingface(&model, 64, false)?;
    engine.load_audio(&model)?;
    for case in info["cases"].as_array().unwrap() {
        let bytes = std::fs::read(case["input"].as_str().unwrap())?;
        let samples: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let features = if info["padding"] == true {
            processor.extract_padded(&samples)?
        } else {
            processor.extract(&samples, Normalization::Dynamic)?
        };
        ensure!(
            features.frames == case["frames"].as_u64().unwrap() as usize,
            "incorrect audio input frame count"
        );
        let actual =
            engine.audio_embeddings(&features, case["chunk_size"].as_u64().unwrap() as usize)?;
        let bytes = std::fs::read(root.join(case["features"].as_str().unwrap()))?;
        ensure!(
            actual.len() * 4 == bytes.len()
                && actual.len() == case["shape"][1].as_u64().unwrap() as usize * 4096,
            "incorrect audio output shape"
        );
        let mut maximum = 0.0f32;
        let mut squared = 0.0f64;
        for (&a, b) in actual.iter().zip(bytes.chunks_exact(4)) {
            let reference = f32::from_le_bytes(b.try_into().unwrap());
            ensure!(
                a.is_finite() && reference.is_finite(),
                "non-finite audio feature"
            );
            let error = a - reference;
            maximum = maximum.max(error.abs());
            squared += f64::from(error).powi(2);
        }
        let rms = (squared / actual.len() as f64).sqrt();
        println!("{}: max_abs={maximum}, rms={rms}", case["name"]);
        ensure!(
            maximum < 0.03 && rms < 0.005,
            "independent Whisper/projector feature mismatch"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires official external padded audio preprocessing fixtures"]
fn ordinary_audio_padding_matches_official_processor() -> Result<()> {
    let root = std::path::PathBuf::from(std::env::var("LIA_AUDIO_PADDED_ORACLE")?);
    let info: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("info.json"))?)?;
    let processor = Processor::default();
    for case in info["cases"].as_array().unwrap() {
        let bytes = std::fs::read(case["input"].as_str().unwrap())?;
        let samples: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let actual = processor.extract_padded(&samples)?;
        ensure!(
            actual.frames == case["frames"].as_u64().unwrap() as usize,
            "incorrect padded audio frame count"
        );
        for (field, tolerance) in [("numpy_mel", 1e-5f32), ("mel", 5e-4)] {
            let bytes = std::fs::read(root.join(case[field].as_str().unwrap()))?;
            ensure!(
                bytes.len() == actual.values.len() * 4,
                "incorrect padded mel extent"
            );
            let mut maximum = 0.0f32;
            for (a, b) in actual.values.iter().zip(bytes.chunks_exact(4)) {
                maximum = maximum.max((a - f32::from_le_bytes(b.try_into().unwrap())).abs());
            }
            println!("{} {field}: max_abs={maximum}", case["name"]);
            ensure!(maximum < tolerance, "padded audio reference mismatch");
        }
    }
    Ok(())
}
