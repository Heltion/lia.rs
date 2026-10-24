//! Read and fold HiFT weight normalization without external inference runtimes.
use super::*;
use crate::torch_weights::Archive;
use anyhow::Context;
impl Engine {
    pub fn load_codec_f0(&mut self, directory: &str) -> Result<()> {
        let checkpoint = crate::models::Checkpoint::open(directory)?;
        ensure!(
            self.geometry.qwen3
                && checkpoint.architecture == crate::models::Architecture::MiniCpmO45,
            "HiFT requires MiniCPM-o 4.5"
        );
        if self
            .weights
            .contains_key("hift.f0_predictor.classifier.weight")
        {
            return Ok(());
        }
        let archive =
            Archive::open(std::path::Path::new(directory).join("assets/token2wav/hift.pt"))?;
        let mut weights = BTreeMap::new();
        for layer in 0..5 {
            let name = format!("f0_predictor.condnet.{}", layer * 2);
            let columns = if layer == 0 { 80 * 3 } else { 512 * 3 };
            let direction = normalized(&archive, &name, &[512, columns / 3, 3])?;
            weights.insert(
                format!("hift.{name}.weight"),
                self.hift_weight(&direction, 512, columns, true)?,
            );
            let bias = values(&archive, &format!("{name}.bias"), &[512])?;
            weights.insert(
                format!("hift.{name}.bias"),
                self.hift_weight(&bias, 1, 512, false)?,
            );
        }
        let name = "f0_predictor.classifier";
        let weight = values(&archive, &format!("{name}.weight"), &[1, 512])?;
        let bias = values(&archive, &format!("{name}.bias"), &[1])?;
        weights.insert(
            format!("hift.{name}.weight"),
            self.hift_weight(&weight, 1, 512, true)?,
        );
        weights.insert(
            format!("hift.{name}.bias"),
            self.hift_weight(&bias, 1, 1, false)?,
        );
        ensure!(
            weights.keys().all(|n| !self.weights.contains_key(n)),
            "HiFT weights overlap engine storage"
        );
        self.weights.extend(weights);
        Ok(())
    }
    pub(super) fn hift_weight(
        &self,
        values: &[f32],
        rows: usize,
        columns: usize,
        matrix: bool,
    ) -> Result<Weight> {
        self.component_f32_weight(values, rows, columns, matrix)
    }
}
pub(super) fn values(archive: &Archive, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    ensure!(
        archive.tensor(name)?.shape == shape,
        "unexpected HiFT shape for {name}"
    );
    let bytes = archive
        .read(name)
        .with_context(|| format!("read HiFT tensor {name}"))?;
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "nonfinite HiFT tensor {name}"
    );
    Ok(values)
}

pub(super) fn normalized(archive: &Archive, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let rows = shape[0];
    let columns = shape[1..].iter().product::<usize>();
    let mut direction = values(
        archive,
        &format!("{name}.parametrizations.weight.original1"),
        shape,
    )?;
    let mut gain_shape = vec![1; shape.len()];
    gain_shape[0] = rows;
    let magnitude = values(
        archive,
        &format!("{name}.parametrizations.weight.original0"),
        &gain_shape,
    )?;
    for (row, gain) in direction.chunks_exact_mut(columns).zip(magnitude) {
        // Preserve the reference Float32 reduction boundary before folding.
        let mut partial = [0.0f32; 256];
        for (lane, value) in partial.iter_mut().enumerate() {
            for &v in row.iter().skip(lane).step_by(256) {
                *value = v.mul_add(v, *value);
            }
        }
        let mut stride = 128;
        while stride > 0 {
            for lane in 0..stride {
                partial[lane] += partial[lane + stride];
            }
            stride /= 2;
        }
        let norm = partial[0].sqrt();
        ensure!(
            norm.is_finite() && norm > 0.,
            "invalid HiFT weight norm for {name}"
        );
        let reciprocal = 1.0 / norm;
        for value in row {
            *value = (gain * *value) * reciprocal;
        }
    }
    Ok(direction)
}
