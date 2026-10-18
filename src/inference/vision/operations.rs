use super::super::*;

impl Engine {
    pub(super) fn vision_mm(
        &self,
        name: &str,
        input: &DeviceBuffer<f32>,
        batch: usize,
    ) -> Result<Buffer> {
        let name = format!("vision.{name}");
        anyhow::ensure!(batch > 0, "empty vision matrix batch");
        let columns = input.len() / batch;
        anyhow::ensure!(
            input.len().is_multiple_of(batch),
            "invalid vision matrix input"
        );
        let padded = self.weight(&name)?.shape[0];
        if columns == padded {
            return self.mm(&name, input);
        }
        anyhow::ensure!(
            columns < padded && padded - columns < 256,
            "invalid padded vision matrix"
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
        self.mm(&name, &aligned)
    }

    pub(super) fn vision_norm(&self, prefix: &str, input: &DeviceBuffer<f32>) -> Result<Buffer> {
        anyhow::ensure!(
            input.len().is_multiple_of(1152),
            "invalid vision normalization input"
        );
        let eps = self
            .vision_eps
            .ok_or_else(|| anyhow!("no vision projector loaded"))?;
        let output = self.zeros(input.len())?;
        // SAFETY: validated row size, complete warps, and separately owned output.
        unsafe {
            self.module.vision_math.layer_norm(
                &self.stream,
                LaunchConfig::for_num_elems((input.len() / 1152 * 32) as u32),
                1152,
                input.len() / 1152,
                eps,
                input,
                self.floats(&format!("vision.{prefix}.weight"))?,
                self.floats(&format!("vision.{prefix}.bias"))?,
                output.cu_deviceptr() as *mut f32,
            )?;
        }
        Ok(output)
    }

    pub(super) fn vision_affine(&self, values: Buffer, name: &str, gelu: bool) -> Result<Buffer> {
        let bias = self.floats(&format!("vision.{name}"))?;
        anyhow::ensure!(
            values.len().is_multiple_of(bias.len()),
            "invalid vision bias shape"
        );
        // SAFETY: the consumed lease uniquely owns all values, and each guarded
        // thread modifies one element using a complete matching bias row.
        unsafe {
            let launch = LaunchConfig::for_num_elems(values.len() as u32);
            let pointer = values.cu_deviceptr() as *mut f32;
            if gelu {
                self.module.vision_math.affine_in_place::<true>(
                    &self.stream,
                    launch,
                    values.len(),
                    bias.len(),
                    bias,
                    pointer,
                )?;
            } else {
                self.module.vision_math.affine_in_place::<false>(
                    &self.stream,
                    launch,
                    values.len(),
                    bias.len(),
                    bias,
                    pointer,
                )?;
            }
        }
        Ok(values)
    }
}
