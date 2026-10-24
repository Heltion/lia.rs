//! Upload model-declared component tensors without embedding architecture names here.
use super::*;

impl Loader<'_> {
    pub(super) fn parameters(
        &self,
        parameters: Vec<crate::models::parameters::Parameter>,
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
                        "invalid BF16 component tensor {}",
                        part.source
                    );
                    let values = self.archive.read(&part.source)?;
                    let range = part.elements.clone().unwrap_or(0..values.len() / 2);
                    ensure!(
                        range.start < range.end && range.end <= values.len() / 2,
                        "invalid component element selection {}",
                        part.source
                    );
                    bytes.extend_from_slice(&values[range.start * 2..range.end * 2]);
                }
                ensure!(
                    bytes.len() == rows * columns * 2,
                    "invalid assembled vision extent {}",
                    parameter.target
                );
                if parameter.transpose {
                    let source = bytes.clone();
                    for row in 0..rows {
                        for col in 0..columns {
                            let dst = (row * columns + col) * 2;
                            let src = (col * rows + row) * 2;
                            bytes[dst..dst + 2].copy_from_slice(&source[src..src + 2]);
                        }
                    }
                }
                let padded = columns.div_ceil(256) * 256;
                let mut data = vec![0u8; rows * padded * 2];
                for row in 0..rows {
                    data[row * padded * 2..(row * padded + columns) * 2]
                        .copy_from_slice(&bytes[row * columns * 2..(row + 1) * columns * 2]);
                }
                Weight {
                    group: None,
                    data: DeviceBuffer::from_host(self.stream, &data).with_context(|| {
                        format!("uploading component matrix {}", parameter.target)
                    })?,
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
                    let source = self.values(&part.source, &part.shape)?;
                    let range = part.elements.clone().unwrap_or(0..source.len());
                    ensure!(
                        range.start < range.end && range.end <= source.len(),
                        "invalid component vector selection"
                    );
                    values.extend_from_slice(&source[range]);
                }
                ensure!(
                    values.len() == parameter.shape.iter().product::<usize>(),
                    "invalid assembled component vector {}",
                    parameter.target
                );
                self.floating_weight(values, parameter.shape.iter().rev().copied().collect())
                    .with_context(|| format!("uploading component vector {}", parameter.target))?
            };
            ensure!(
                weights.insert(parameter.target, weight).is_none(),
                "duplicate model component target"
            );
        }
        Ok(())
    }
}
