use super::*;

impl Loader<'_> {
    pub(super) fn values(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let tensor = self.archive.tensor(name)?;
        ensure!(tensor.shape == shape, "unexpected shape for {name}");
        let bytes = self.archive.read(name)?;
        let values = match tensor.dtype {
            Dtype::Bf16 => bytes
                .chunks_exact(2)
                .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
                .collect::<Vec<_>>(),
            Dtype::F32 => bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            _ => anyhow::bail!("unsupported floating tensor dtype for {name}"),
        };
        ensure!(
            values.iter().all(|x| x.is_finite()),
            "nonfinite tensor {name}"
        );
        Ok(super::layout::rows(name, values, shape[0]))
    }

    pub(super) fn floating_weight(&self, values: Vec<f32>, shape: Vec<usize>) -> Result<Weight> {
        let bytes = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        Ok(Weight {
            group: None,
            data: DeviceBuffer::from_host(self.stream, &bytes)?,
            floats: Some(DeviceBuffer::from_host(self.stream, &values)?),
            shape,
            kind: 0,
            fp8_scale: None,
            block: 1,
            bytes: 4,
        })
    }

    pub(super) fn floats(
        &self,
        name: &str,
        shape: &[usize],
        transform: fn(f32) -> f32,
    ) -> Result<Weight> {
        let values = self
            .values(name, shape)?
            .into_iter()
            .map(transform)
            .collect::<Vec<_>>();
        ensure!(
            values.iter().all(|x| x.is_finite()),
            "nonfinite transformed tensor {name}"
        );
        self.floating_weight(values, shape.iter().rev().copied().collect())
    }

    pub(super) fn dense(&self, name: &str, shape: &[usize]) -> Result<Weight> {
        let tensor = self.archive.tensor(name)?;
        ensure!(
            tensor.dtype == Dtype::Bf16 && tensor.shape == shape,
            "invalid BF16 tensor {name}"
        );
        let mut bytes = super::layout::rows(name, self.archive.read(name)?, shape[0]);
        super::layout::dense_columns(name, &mut bytes, shape)?;
        let data = if name == "model.language_model.embed_tokens.weight"
            && std::env::var("LIA_HOST_EMBEDDING").as_deref() == Ok("1")
        {
            crate::inference::host_weights::allocate(self.stream.context(), &bytes)?
        } else {
            DeviceBuffer::from_host(self.stream, &bytes)?
        };
        Ok(Weight {
            group: None,
            data,
            floats: None,
            shape: shape.iter().rev().copied().collect(),
            kind: 30,
            fp8_scale: None,
            block: 1,
            bytes: 2,
        })
    }

    pub(super) fn quant(
        &self,
        name: &str,
        bits: u32,
        group: usize,
        rows: usize,
        columns: usize,
    ) -> Result<Weight> {
        let mut matrix = Matrix::load(self.archive, name, bits, group)?;
        super::layout::quant(name, &mut matrix);
        ensure!(
            (matrix.rows, matrix.columns) == (rows, columns),
            "invalid grouped matrix dimensions for {name}"
        );
        Ok(Weight {
            data: DeviceBuffer::from_host(self.stream, &matrix.packed)?,
            floats: None,
            shape: vec![columns, rows],
            kind: u32::MAX,
            fp8_scale: None,
            block: 1,
            bytes: 1,
            group: Some(GroupScale {
                bits,
                size: matrix.group_size,
                scales: DeviceBuffer::from_host(self.stream, &matrix.scales)?,
            }),
        })
    }

    pub(super) fn expert_stack(
        &self,
        prefix: &str,
        projection: &str,
        rows: usize,
        columns: usize,
    ) -> Result<Matrix> {
        let mut packed = Vec::with_capacity(256 * rows * columns / (8 / self.expert_bits as usize));
        let mut scales = Vec::with_capacity(256 * rows * columns / 128);
        for expert in 0..256 {
            let matrix = Matrix::load(
                self.archive,
                &format!("{prefix}.{expert}.{projection}"),
                self.expert_bits,
                128,
            )?;
            ensure!(
                (matrix.rows, matrix.columns) == (rows, columns),
                "invalid expert dimensions"
            );
            packed.extend_from_slice(&matrix.packed);
            scales.extend_from_slice(&matrix.scales);
        }
        Ok(Matrix {
            rows: 256 * rows,
            columns,
            bits: self.expert_bits,
            group_size: 128,
            packed,
            scales,
        })
    }

    pub(super) fn language_layer(
        &self,
        layer: usize,
        weights: &mut BTreeMap<String, Weight>,
    ) -> Result<()> {
        use crate::models::qwen35::weights::{self as mapping, Storage};
        for (target, parameter) in mapping::layer(self.architecture, layer)? {
            let shape = &parameter.shape;
            let weight = match parameter.storage {
                Storage::Dense => self.dense(&parameter.source, shape)?,
                Storage::Grouped { bits, group } => {
                    self.quant(&parameter.source, bits, group, shape[0], shape[1])?
                }
                Storage::Floating(transform) => self.floats(&parameter.source, shape, transform)?,
            };
            weights.insert(target, weight);
        }
        Ok(())
    }
}
