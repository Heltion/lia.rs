//! Exact, stream-ordered page transfers preserve every layer and KV plane.
use super::*;
use anyhow::ensure;
pub const MINICPM5_PAGE_BYTES: usize = 42 * 2 * 32 * 256 * 2;
const PLANE_BYTES: usize = 32 * 256 * 2;

impl Engine {
    pub fn kv_transfer_backend(&self) -> &'static str {
        if self
            .paging
            .as_ref()
            .is_some_and(|work| work.upload.is_some())
        {
            if self.kv_batch_copy.get() {
                "cuda_pinned_upload_batch_download"
            } else {
                "cuda_pinned_upload_plane_download"
            }
        } else if self.kv_batch_copy.get() {
            "cuda_batch_copy"
        } else {
            "cuda_plane_copies"
        }
    }

    pub fn read_kv_page(&self, page: u32) -> Result<Vec<u8>> {
        self.validate_kv_page(page)?;
        let mut bytes = vec![0_u8; MINICPM5_PAGE_BYTES];
        self.transfer_kv_pages_raw(&[(page, bytes.as_mut_ptr())], false)?;
        Ok(bytes)
    }

    pub fn write_kv_page(&mut self, page: u32, bytes: &[u8]) -> Result<()> {
        self.write_kv_pages(&[(page, bytes)])
    }

    pub fn write_kv_pages(&mut self, pages: &[(u32, &[u8])]) -> Result<()> {
        ensure!(
            !pages.is_empty() && pages.len() <= 16,
            "invalid KV transfer window"
        );
        let mut seen = std::collections::BTreeSet::new();
        let mut pointers = Vec::with_capacity(pages.len());
        for &(page, bytes) in pages {
            self.validate_kv_page(page)?;
            ensure!(
                bytes.len() == MINICPM5_PAGE_BYTES,
                "invalid saved KV page extent"
            );
            ensure!(seen.insert(page), "duplicate KV transfer destination");
            pointers.push((page, bytes.as_ptr() as *mut u8));
        }
        if let Some(upload) = self.paging.as_mut().and_then(|work| work.upload.as_mut()) {
            return upload.write(&self.module, &self.stream, &self.states, pages);
        }
        // Upload only reads caller allocations, retaining them through synchronization.
        self.transfer_kv_pages_raw(&pointers, true)
    }

    fn validate_kv_page(&self, page: u32) -> Result<()> {
        ensure!(
            self.geometry.llama && self.paging.is_some() && self.states.len() == 42,
            "KV pages require configured MiniCPM5 paging"
        );
        ensure!(
            !self.kv_offloaded,
            "paged KV is unavailable while offloaded"
        );
        ensure!(
            (page as usize) < self.capacity / 32,
            "KV page outside physical pool"
        );
        Ok(())
    }

    fn transfer_kv_pages_raw(&self, pages: &[(u32, *mut u8)], upload: bool) -> Result<()> {
        let count = pages.len() * 84;
        let mut sources = vec![0_u64; count];
        let mut destinations = vec![0_u64; count];
        let mut sizes = vec![PLANE_BYTES; count];
        for (block, &(page, host)) in pages.iter().enumerate() {
            for (layer, state) in self.states.iter().enumerate() {
                let State::Attention { k, v } = state else {
                    anyhow::bail!("unexpected non-attention KV layer");
                };
                for (plane, device) in [k, v].iter().enumerate() {
                    ensure!(
                        device.len() >= (page as usize + 1) * PLANE_BYTES / 2,
                        "KV layer {layer} plane {plane} is unavailable"
                    );
                    let index = block * 84 + layer * 2 + plane;
                    let address = device.cu_deviceptr() + page as u64 * PLANE_BYTES as u64;
                    // SAFETY: each retained host page covers all84 complete planes.
                    let pointer = unsafe { host.add((layer * 2 + plane) * PLANE_BYTES) } as u64;
                    (sources[index], destinations[index]) = if upload {
                        (pointer, address)
                    } else {
                        (address, pointer)
                    };
                }
            }
        }
        if self.stream.cu_stream().is_null() {
            self.kv_batch_copy.set(false);
        }
        if self.kv_batch_copy.get() {
            // Zero location hints/flags mean no preference. Every plain C field permits zero.
            let mut attributes: cuda_core::sys::CUmemcpyAttributes = unsafe { std::mem::zeroed() };
            attributes.srcAccessOrder = if upload {
                // Immutable host data has no earlier producer on this stream.
                cuda_core::sys::CUmemcpySrcAccessOrder_enum_CU_MEMCPY_SRC_ACCESS_ORDER_ANY
            } else {
                // Device KV must observe earlier model writes.
                cuda_core::sys::CUmemcpySrcAccessOrder_enum_CU_MEMCPY_SRC_ACCESS_ORDER_STREAM
            };
            let mut first = 0_usize;
            // SAFETY: all planes have validated extents and disjoint destinations. Arrays and
            // host data remain live through synchronization, even after enqueue failure.
            let status = unsafe {
                cuda_core::sys::cuMemcpyBatchAsync_v2(
                    destinations.as_mut_ptr(),
                    sources.as_mut_ptr(),
                    sizes.as_mut_ptr(),
                    count,
                    &mut attributes,
                    &mut first,
                    1,
                    self.stream.cu_stream(),
                )
            };
            let synchronized = self.stream.synchronize();
            synchronized?;
            if status == 0 {
                return Ok(());
            }
            ensure!(
                status == 1 || status == 500 || status == 801,
                "batching KV page transfers: {:?}",
                cuda_core::DriverError(status)
            );
            // Older drivers or unsupported pageable copies retain the exact plane-copy path.
            self.kv_batch_copy.set(false);
        }
        let result = (|| -> Result<()> {
            for index in 0..count {
                let status = unsafe {
                    if upload {
                        cuda_core::sys::cuMemcpyHtoDAsync_v2(
                            destinations[index],
                            sources[index] as *const _,
                            PLANE_BYTES,
                            self.stream.cu_stream(),
                        )
                    } else {
                        cuda_core::sys::cuMemcpyDtoHAsync_v2(
                            destinations[index] as *mut _,
                            sources[index],
                            PLANE_BYTES,
                            self.stream.cu_stream(),
                        )
                    }
                };
                ensure!(
                    status == 0,
                    "transferring KV layer {} plane {}: {:?}",
                    index % 84 / 2,
                    index % 2,
                    cuda_core::DriverError(status)
                );
            }
            Ok(())
        })();
        let synchronized = self.stream.synchronize();
        result?;
        synchronized?;
        Ok(())
    }
}
