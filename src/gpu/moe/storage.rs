//! Resident expert prefix and immutable, page-locked host suffix.
use super::*;
use cuda_core::{PinnedHostBuffer, simt::device_buffer::DeviceCopy};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Layout {
    RowMajor,
    Register,
}

pub(super) struct Packed {
    layout: Layout,
    register_compatible: bool,
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
            layout: Layout::RowMajor,
            register_compatible: scales.iter().all(|s| (1..=253).contains(&((s >> 7) & 255))),
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

    pub(super) fn repacked(
        &self,
        module: &kernels::LoadedModule,
        stream: &std::sync::Arc<CudaStream>,
        rows: usize,
        columns: usize,
    ) -> Result<Self> {
        ensure!(
            self.can_register(),
            "register MMA requires resident normal-scale row-major experts"
        );
        ensure!(
            rows.is_multiple_of(16)
                && columns.is_multiple_of(128)
                && self.data.len() == rows * columns / 4
                && self.scales.len() == rows * columns / 128,
            "invalid register MMA repack extent"
        );
        let data = DeviceBuffer::zeroed(stream, self.data.len())?;
        let scales = DeviceBuffer::zeroed(stream, self.scales.len())?;
        // SAFETY: complete tiles and disjoint equally sized allocations. CUDA
        // buffer alignment permits the u16 view of the packed byte storage.
        unsafe {
            module.moe_repack.weights(
                stream,
                LaunchConfig::for_num_elems((data.len() / 2) as u32),
                rows,
                columns,
                &self.data,
                data.cu_deviceptr() as *mut u16,
            )?;
            module.moe_repack.scales(
                stream,
                LaunchConfig::for_num_elems(scales.len() as u32),
                rows,
                columns,
                &self.scales,
                scales.cu_deviceptr() as *mut u16,
            )?;
        }
        Ok(Self {
            layout: Layout::Register,
            register_compatible: true,
            data,
            scales,
            host_data: None,
            host_scales: None,
        })
    }

    pub fn can_register(&self) -> bool {
        self.layout == Layout::RowMajor && self.register_compatible && !self.is_offloaded()
    }

    pub fn is_register(&self) -> bool {
        self.layout == Layout::Register
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
