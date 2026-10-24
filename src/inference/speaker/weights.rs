//! Load the checked embedded ONNX Float32 tensors directly into native GPU storage.
use super::*;
impl Engine {
    pub fn load_speaker(&mut self, directory: &str) -> Result<()> {
        let checkpoint = crate::models::Checkpoint::open(directory)?;
        ensure!(
            self.geometry.qwen3
                && checkpoint.architecture == crate::models::Architecture::MiniCpmO45,
            "speaker encoder requires MiniCPM-o 4.5"
        );
        if self
            .weights
            .contains_key("speaker.xvector.dense.linear.weight")
        {
            return Ok(());
        }
        let archive = crate::onnx_weights::Archive::open(
            std::path::Path::new(directory).join("assets/token2wav/campplus.onnx"),
        )?;
        let plan = crate::models::minicpm_o45::speaker::plan(&archive)?;
        let mut weights = BTreeMap::new();
        for parameter in plan {
            ensure!(
                parameter.parts.len() == 1 && !parameter.transpose,
                "invalid speaker assembly plan"
            );
            let values = archive.floats(&parameter.parts[0].source)?;
            let (rows, columns) = if parameter.matrix {
                (parameter.shape[0], parameter.shape[1])
            } else {
                (1, parameter.shape[0])
            };
            ensure!(
                weights
                    .insert(
                        parameter.target,
                        self.component_f32_weight(&values, rows, columns, parameter.matrix)?
                    )
                    .is_none(),
                "duplicate speaker parameter"
            );
        }
        ensure!(
            weights.keys().all(|n| !self.weights.contains_key(n)),
            "speaker parameter overlap"
        );
        self.weights.extend(weights);
        Ok(())
    }
}
