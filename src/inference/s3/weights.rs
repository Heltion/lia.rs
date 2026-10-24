//! Checked native loading of S3's exported ONNX parameters.
use super::*;
use std::collections::BTreeMap;
impl Engine {
    pub fn load_s3(&mut self, directory: &str) -> Result<()> {
        let checkpoint = crate::models::Checkpoint::open(directory)?;
        ensure!(
            self.geometry.qwen3
                && checkpoint.architecture == crate::models::Architecture::MiniCpmO45,
            "S3 requires MiniCPM-o 4.5"
        );
        if self.weights.contains_key("s3.quantizer.weight") {
            return Ok(());
        }
        let archive = crate::onnx_weights::Archive::open(
            std::path::Path::new(directory).join("assets/token2wav/speech_tokenizer_v2_25hz.onnx"),
        )?;
        let plan = crate::models::minicpm_o45::s3_weights::plan(&archive)?;
        let mut weights = BTreeMap::new();
        for parameter in plan {
            ensure!(parameter.parts.len() == 1, "invalid S3 assembly plan");
            let original = archive.floats(&parameter.parts[0].source)?;
            let values = if parameter.transpose {
                let rows = parameter.shape[0];
                let columns = parameter.shape[1];
                let mut values = vec![0.; original.len()];
                for row in 0..rows {
                    for col in 0..columns {
                        values[row * columns + col] = original[col * rows + row];
                    }
                }
                values
            } else {
                original
            };
            let (rows, columns) = if parameter.matrix {
                (parameter.shape[0], parameter.shape[1])
            } else {
                (1, values.len())
            };
            ensure!(
                weights
                    .insert(
                        parameter.target,
                        self.component_f32_weight(&values, rows, columns, parameter.matrix)?
                    )
                    .is_none(),
                "duplicate S3 parameter"
            );
        }
        ensure!(
            weights.keys().all(|n| !self.weights.contains_key(n)),
            "S3 parameter overlap"
        );
        self.weights.extend(weights);
        Ok(())
    }
}
