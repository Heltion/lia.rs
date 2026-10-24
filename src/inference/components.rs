//! Dense component projections share padding and compensated activation products.
use super::*;

impl Engine {
    pub(super) fn component_f32_weight(
        &self,
        values: &[f32],
        rows: usize,
        columns: usize,
        matrix: bool,
    ) -> Result<Weight> {
        anyhow::ensure!(
            rows.checked_mul(columns) == Some(values.len()) && values.iter().all(|v| v.is_finite()),
            "invalid component weight dimensions or values"
        );
        let padded = if matrix {
            columns.div_ceil(256) * 256
        } else {
            columns
        };
        let mut bytes = vec![0u8; rows * padded * 4];
        for row in 0..rows {
            for col in 0..columns {
                bytes[(row * padded + col) * 4..(row * padded + col + 1) * 4]
                    .copy_from_slice(&values[row * columns + col].to_le_bytes());
            }
        }
        Ok(Weight {
            group: None,
            data: DeviceBuffer::from_host(&self.stream, &bytes)?,
            floats: if matrix {
                None
            } else {
                Some(DeviceBuffer::from_host(&self.stream, values)?)
            },
            shape: vec![padded, rows],
            kind: 0,
            fp8_scale: None,
            block: 1,
            bytes: 4,
        })
    }
    pub(super) fn component_mm(
        &self,
        name: &str,
        input: &DeviceBuffer<f32>,
        batch: usize,
        compensate: bool,
    ) -> Result<Buffer> {
        anyhow::ensure!(batch > 0, "empty component matrix batch");
        let columns = input.len() / batch;
        anyhow::ensure!(
            input.len().is_multiple_of(batch),
            "invalid component matrix input"
        );
        let padded = self.weight(name)?.shape[0];
        if columns == padded {
            return self.component_product(name, input, compensate);
        }
        anyhow::ensure!(
            columns < padded && padded - columns < 256,
            "invalid padded component matrix"
        );
        let mut aligned = self.zeros(padded * batch)?;
        // SAFETY: input covers batch complete logical rows; destination covers
        // batch complete padded rows, and zero tails never read input.
        unsafe {
            self.module.vision.pad_rows(
                &self.stream,
                LaunchConfig::for_num_elems(aligned.len() as u32),
                columns,
                padded,
                input,
                &mut aligned,
            )?;
        }
        self.component_product(name, &aligned, compensate)
    }

    fn component_product(
        &self,
        name: &str,
        input: &DeviceBuffer<f32>,
        compensate: bool,
    ) -> Result<Buffer> {
        let weight = self.weight(name)?;
        let batch = input.len() / weight.shape[0];
        if compensate && weight.kind == 0 && batch > 8 {
            let mut output = self.zeros(batch * weight.shape[1])?;
            crate::gpu::float_matrix::launch(
                &self.module,
                &self.stream,
                weight.shape[0],
                weight.shape[1],
                batch,
                &weight.data,
                input,
                &mut output,
            )?;
            return Ok(output);
        }
        let high = self.mm(name, input)?;
        // Small BF16/F32 projections accumulate FP32 activations directly.
        if !compensate || batch <= 8 || (weight.kind == 30 && weight.shape[1] <= 96) {
            return Ok(high);
        }
        // Two activation products retain the FP32 residual through the encoder
        // instead of discarding it at every FP16 matrix product.
        let mut residual = self.zeros(input.len())?;
        // SAFETY: one disjoint output for each input value on the same stream.
        unsafe {
            self.module.minicpm_vision.minicpm_f16_residual(
                &self.stream,
                LaunchConfig::for_num_elems(input.len() as u32),
                input,
                &mut residual,
            )?;
        }
        let low = self.mm(name, &residual)?;
        self.add(&high, &low)
    }
}
