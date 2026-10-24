//! Speech decoder weights and checkpoint weight-normalized audio-code head.
use super::*;

impl Engine {
    pub fn load_speech(&mut self, directory: &str) -> Result<()> {
        let checkpoint = crate::models::Checkpoint::open(directory)?;
        ensure!(
            self.geometry.qwen3
                && checkpoint.architecture == crate::models::Architecture::MiniCpmO45,
            "speech decoder requires MiniCPM-o 4.5"
        );
        if self.weights.contains_key("speech.head") {
            return Ok(());
        }
        let config = &checkpoint.config["tts_config"];
        ensure!(
            config["backbone_model"] == "llama"
                && config["condition_type"] == "hidden_text_merge"
                && config["projector_type"] == "mlp"
                && config["normalize_projected_hidden"] == true
                && config["num_vq"] == 1
                && config["attention_type"] == "full_attention",
            "unsupported speech decoder conditioning or attention"
        );
        let archive = Archive::open(directory)?;
        let loader = Loader {
            archive: &archive,
            architecture: checkpoint.architecture,
            expert_bits: 0,
            stream: &self.stream,
        };
        let mut weights = BTreeMap::new();
        loader.parameters(
            crate::models::minicpm_o45::speech_weights::plan(),
            &mut weights,
        )?;
        let magnitude = loader.values(
            "tts.head_code.0.parametrizations.weight.original0",
            &[6562, 1],
        )?;
        let mut direction = loader.values(
            "tts.head_code.0.parametrizations.weight.original1",
            &[6562, 768],
        )?;
        for (row, magnitude) in direction.chunks_exact_mut(768).zip(magnitude) {
            let norm = row
                .iter()
                .map(|v| f64::from(*v).powi(2))
                .sum::<f64>()
                .sqrt();
            ensure!(
                norm.is_finite() && norm > 0.0,
                "invalid speech head weight norm"
            );
            for value in row {
                *value = (f64::from(*value) * f64::from(magnitude) / norm) as f32;
            }
        }
        weights.insert(
            "speech.head".into(),
            loader.floating_weight(direction, vec![768, 6562])?,
        );
        ensure!(
            weights.keys().all(|name| !self.weights.contains_key(name)),
            "duplicate speech parameter"
        );
        self.weights.extend(weights);
        Ok(())
    }
}
