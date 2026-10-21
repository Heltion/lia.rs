//! Upload model-declared vision tensors without embedding architecture names here.
use super::*;

impl Loader<'_> {
    pub(super) fn vision_parameters(
        &self,
        parameters: Vec<crate::models::vision::Parameter>,
        weights: &mut BTreeMap<String, Weight>,
    ) -> Result<()> {
        for parameter in parameters {
            let weight = if parameter.matrix {
                let (rows, columns) = (parameter.shape[0], parameter.shape[1]);
                let mut bytes = Vec::new();
                for part in &parameter.parts {
                    let tensor = self.archive.tensor(&part.source)?;
                    ensure!(
                        tensor.dtype == Dtype::Bf16 && tensor.shape == part.shape,
                        "invalid BF16 vision tensor {}",
                        part.source
                    );
                    bytes.extend_from_slice(&self.archive.read(&part.source)?);
                }
                ensure!(
                    bytes.len() == rows * columns * 2,
                    "invalid assembled vision extent {}",
                    parameter.target
                );
                let padded = columns.div_ceil(256) * 256;
                let mut data = vec![0u8; rows * padded * 2];
                for row in 0..rows {
                    data[row * padded * 2..(row * padded + columns) * 2]
                        .copy_from_slice(&bytes[row * columns * 2..(row + 1) * columns * 2]);
                }
                Weight {
                    group: None,
                    data: DeviceBuffer::from_host(self.stream, &data)
                        .with_context(|| format!("uploading vision matrix {}", parameter.target))?,
                    floats: None,
                    shape: vec![padded, rows],
                    kind: 30,
                    fp8_scale: None,
                    block: 1,
                    bytes: 2,
                }
            } else {
                let mut values = Vec::new();
                for part in &parameter.parts {
                    values.extend(self.values(&part.source, &part.shape)?);
                }
                ensure!(
                    values.len() == parameter.shape.iter().product::<usize>(),
                    "invalid assembled vision vector {}",
                    parameter.target
                );
                self.floating_weight(values, parameter.shape.iter().rev().copied().collect())
                    .with_context(|| format!("uploading vision vector {}", parameter.target))?
            };
            ensure!(
                weights.insert(parameter.target, weight).is_none(),
                "duplicate model vision target"
            );
        }
        Ok(())
    }
}
