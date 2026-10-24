//! Checked Float32 codec assembly; weights remain in the Rust CUDA engine.
use super::*;
use crate::models::parameters::Parameter;
use anyhow::{Context, ensure};

impl Engine {
    pub fn load_codec_encoder(&mut self, directory: &str) -> Result<()> {
        ensure!(self.geometry.qwen3, "codec encoder requires MiniCPM-o");
        let checkpoint = crate::models::Checkpoint::open(directory)?;
        ensure!(
            checkpoint.architecture == crate::models::Architecture::MiniCpmO45,
            "codec checkpoint must be MiniCPM-o 4.5"
        );
        if self.weights.contains_key("codec.encoder_proj.weight") {
            return Ok(());
        }
        let archive = crate::torch_weights::Archive::open(
            std::path::Path::new(directory).join("assets/token2wav/flow.pt"),
        )?;
        let mut weights = BTreeMap::new();
        for parameter in crate::models::minicpm_o45::codec_weights::encoder_plan() {
            let weight = self.codec_parameter(&archive, &parameter)?;
            ensure!(
                weights.insert(parameter.target, weight).is_none(),
                "duplicate codec parameter"
            );
        }
        ensure!(
            weights.keys().all(|name| !self.weights.contains_key(name)),
            "codec parameters already overlap engine storage"
        );
        self.weights.extend(weights);
        Ok(())
    }
    pub fn load_codec_flow(&mut self, directory: &str) -> Result<()> {
        ensure!(self.geometry.qwen3, "flow decoder requires MiniCPM-o");
        let checkpoint = crate::models::Checkpoint::open(directory)?;
        ensure!(
            checkpoint.architecture == crate::models::Architecture::MiniCpmO45,
            "flow checkpoint must be MiniCPM-o 4.5"
        );
        if self
            .weights
            .contains_key("codec.decoder.estimator.final_layer.linear.weight")
        {
            return Ok(());
        }
        let archive = crate::torch_weights::Archive::open(
            std::path::Path::new(directory).join("assets/token2wav/flow.pt"),
        )?;
        let mut weights = BTreeMap::new();
        for parameter in crate::models::minicpm_o45::cfm_weights::plan() {
            let weight = self.codec_parameter(&archive, &parameter)?;
            ensure!(
                weights.insert(parameter.target, weight).is_none(),
                "duplicate flow parameter"
            );
        }
        ensure!(
            weights.keys().all(|name| !self.weights.contains_key(name)),
            "flow parameters overlap engine storage"
        );
        self.weights.extend(weights);
        Ok(())
    }

    fn codec_parameter(
        &self,
        archive: &crate::torch_weights::Archive,
        parameter: &Parameter,
    ) -> Result<Weight> {
        let mut bytes = Vec::new();
        for part in &parameter.parts {
            let tensor = archive.tensor(&part.source)?;
            ensure!(
                tensor.shape == part.shape && part.elements.is_none(),
                "invalid codec source {}",
                part.source
            );
            bytes.extend(archive.read(&part.source)?);
        }
        let count = parameter
            .shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .context("codec extent overflow")?;
        ensure!(
            bytes.len() == count * 4 && !parameter.transpose,
            "invalid codec assembled extent"
        );
        ensure!(
            bytes
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
            "nonfinite codec parameter {}",
            parameter.target
        );
        let (data, floats, shape) = if parameter.matrix {
            let rows = parameter.shape[0];
            let columns = parameter.shape[1];
            let padded = columns.div_ceil(256) * 256;
            let mut data = vec![0u8; rows * padded * 4];
            for row in 0..rows {
                data[row * padded * 4..(row * padded + columns) * 4]
                    .copy_from_slice(&bytes[row * columns * 4..(row + 1) * columns * 4]);
            }
            (
                DeviceBuffer::from_host(&self.stream, &data)?,
                None,
                vec![padded, rows],
            )
        } else {
            let values: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            (
                DeviceBuffer::from_host(&self.stream, &bytes)?,
                Some(DeviceBuffer::from_host(&self.stream, &values)?),
                parameter.shape.iter().rev().copied().collect(),
            )
        };
        Ok(Weight {
            group: None,
            data,
            floats,
            shape,
            kind: 0,
            fp8_scale: None,
            block: 1,
            bytes: 4,
        })
    }
}
