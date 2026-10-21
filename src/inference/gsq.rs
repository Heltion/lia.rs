use super::*;

impl Engine {
    pub(super) fn gsq_mm(
        &self,
        weight: &Weight,
        input: &DeviceBuffer<f32>,
        reuse_input: bool,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let group = weight
            .group
            .as_ref()
            .ok_or_else(|| anyhow!("missing GSQ scales"))?;
        let columns = weight.shape[0];
        let rows = weight.shape[1];
        if input.len() / columns <= 8 {
            return crate::gpu::gsq::matmul(
                &self.module,
                &self.stream,
                group.bits,
                group.size,
                columns,
                rows,
                &weight.data,
                &group.scales,
                input,
                output,
            );
        }
        let mut packed = self.prefill_input.borrow_mut();
        if packed.as_ref().is_none_or(|p| p.len() < input.len() / 2) {
            anyhow::ensure!(!self.preallocated, "reserved GSQ input extent exceeded");
            *packed = Some(DeviceBuffer::zeroed(&self.stream, input.len() / 2)?);
        }
        crate::gpu::gsq_tensor::matmul(
            &self.module,
            &self.stream,
            group.bits,
            group.size,
            columns,
            rows,
            &weight.data,
            &group.scales,
            input,
            packed
                .as_ref()
                .ok_or_else(|| anyhow!("missing GSQ input workspace"))?,
            reuse_input,
            output,
        )
    }
}
