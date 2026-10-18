//! Resident expert prefix and immutable, page-locked host suffix.
use super::*;
use cuda_core::{PinnedHostBuffer, simt::device_buffer::DeviceCopy};

pub(super) struct Packed {
    pub data: DeviceBuffer<u8>,
    pub scales: DeviceBuffer<u16>,
    host_data: Option<Mapped<u8>>,
    host_scales: Option<Mapped<u16>>,
}

struct Mapped<T: DeviceCopy> {
    // The owner must outlive every kernel and graph referencing device_ptr.
    _owner: PinnedHostBuffer<T>,
    device_ptr: u64,
}

impl<T: DeviceCopy> Mapped<T> {
    fn new(stream: &std::sync::Arc<CudaStream>, values: &[T]) -> Result<Self> {
        let owner = PinnedHostBuffer::from_slice(stream.context(), values)?;
        let mut device_ptr = 0;
        // SAFETY: CUDA owns this live page-locked allocation in the bound
        // context. The returned mapping is borrowed until owner is dropped.
        let status = unsafe {
            cuda_core::sys::cuMemHostGetDevicePointer_v2(
                &mut device_ptr,
                owner.as_ptr().cast_mut().cast(),
                0,
            )
        };
        ensure!(
            status == 0,
            "mapping host expert storage: {:?}",
            cuda_core::DriverError(status)
        );
        Ok(Self {
            _owner: owner,
            device_ptr,
        })
    }
}

impl Packed {
    pub fn upload(
        stream: &std::sync::Arc<CudaStream>,
        matrix: Matrix,
        rows: usize,
        columns: usize,
        offload: usize,
    ) -> Result<Self> {
        ensure!(
            matrix.rows == rows
                && matrix.columns == columns
                && matrix.bits == 2
                && matrix.group_size == 128
                && matrix.packed.len() == rows * columns / 4
                && matrix.scales.len() == rows * columns / 128,
            "invalid packed expert stack"
        );
        ensure!(
            matrix
                .scales
                .iter()
                .all(|s| s.is_finite() && s.to_bits() & 0xffff == 0),
            "expert scales must be exactly representable as checkpoint BF16"
        );
        let scales = matrix
            .scales
            .iter()
            .map(|s| (s.to_bits() >> 16) as u16)
            .collect::<Vec<_>>();
        let resident_rows = rows / 256 * (256 - offload);
        let data_split = resident_rows * columns / 4;
        let scale_split = resident_rows * columns / 128;
        Ok(Self {
            data: DeviceBuffer::from_host(stream, &matrix.packed[..data_split])?,
            scales: DeviceBuffer::from_host(stream, &scales[..scale_split])?,
            host_data: if offload == 0 {
                None
            } else {
                Some(Mapped::new(stream, &matrix.packed[data_split..])?)
            },
            host_scales: if offload == 0 {
                None
            } else {
                Some(Mapped::new(stream, &scales[scale_split..])?)
            },
        })
    }

    pub fn is_offloaded(&self) -> bool {
        self.host_data.is_some()
    }

    pub fn host_data(&self) -> *const u8 {
        self.host_data
            .as_ref()
            .map_or(std::ptr::null(), |v| v.device_ptr as *const u8)
    }

    pub fn host_scales(&self) -> *const u16 {
        self.host_scales
            .as_ref()
            .map_or(std::ptr::null(), |v| v.device_ptr as *const u16)
    }
}
