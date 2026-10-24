//! Audio weights are optional and share the owning language engine's CUDA stream.
use super::*;

impl Engine {
    pub fn load_audio(&mut self, directory: &str) -> Result<()> {
        let checkpoint = crate::models::Checkpoint::open(directory)?;
        ensure!(
            self.geometry.qwen3
                && checkpoint.architecture == crate::models::Architecture::MiniCpmO45,
            "audio encoder requires the MiniCPM-o 4.5 engine"
        );
        if self.weights.contains_key("audio.conv1.weight") {
            return Ok(());
        }
        let archive = Archive::open(directory)?;
        let loader = Loader {
            archive: &archive,
            architecture: checkpoint.architecture,
            expert_bits: 0,
            stream: &self.stream,
        };
        let mut weights = BTreeMap::new();
        loader.parameters(
            crate::models::minicpm_o45::audio_weights::plan(),
            &mut weights,
        )?;
        ensure!(
            weights.keys().all(|name| !self.weights.contains_key(name)),
            "duplicate audio parameter"
        );
        self.weights.extend(weights);
        Ok(())
    }
}
