//! Reuse one reserved device suffix for the current layer's host expert weights.
use super::*;

pub(super) struct HostStage {
    data: DeviceBuffer<u8>,
    scales: DeviceBuffer<u16>,
}

impl HostStage {
    pub(super) fn new(stream: &std::sync::Arc<CudaStream>, experts: &Experts) -> Result<Self> {
        let (gate, gs) = experts.gate.host_extents();
        let (up, us) = experts.up.host_extents();
        let (down, ds) = experts.down.host_extents();
        Ok(Self {
            data: DeviceBuffer::zeroed(stream, (gate + up).max(down))?,
            scales: DeviceBuffer::zeroed(stream, (gs + us).max(ds))?,
        })
    }

    pub(super) fn upload(
        &self,
        stream: &CudaStream,
        weights: &Packed,
        up: &Packed,
        gate: bool,
    ) -> Result<(*const u8, *const u16, *const u8, *const u16)> {
        let (bytes, scales) = weights.host_extents();
        let (up_bytes, up_scales) = if gate { up.host_extents() } else { (0, 0) };
        ensure!(
            bytes + up_bytes <= self.data.len() && scales + up_scales <= self.scales.len(),
            "reserved expert staging extent exceeded"
        );
        let data = self.data.cu_deviceptr();
        let scale = self.scales.cu_deviceptr();
        weights.stage_host(stream, data, scale)?;
        let (upper, upper_scale) = if gate {
            let upper = data + bytes as u64;
            let upper_scale = scale + (scales * 2) as u64;
            up.stage_host(stream, upper, upper_scale)?;
            (upper, upper_scale)
        } else {
            (data, scale)
        };
        Ok((
            data as *const u8,
            scale as *const u16,
            upper as *const u8,
            upper_scale as *const u16,
        ))
    }
}
