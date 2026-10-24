//! Bounded pinned staging combines page payloads into one host-to-device copy.
use super::*;
use anyhow::{Context, ensure};
const HEADER: usize = 1024;
const PAGE: usize = super::paging_transfer::MINICPM5_PAGE_BYTES;

pub(super) struct Upload {
    host: cuda_core::PinnedHostBuffer<u8>,
    device: DeviceBuffer<u8>,
}

impl Upload {
    pub fn new(context: &Arc<CudaContext>, stream: &CudaStream) -> Result<Self> {
        let bytes = HEADER + 16 * PAGE;
        Ok(Self {
            host: cuda_core::PinnedHostBuffer::zeroed(context, bytes)
                .context("reserving pinned KV upload window")?,
            device: DeviceBuffer::zeroed(stream, bytes)
                .context("reserving GPU KV upload window")?,
        })
    }

    pub fn write(
        &mut self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        states: &[State],
        pages: &[(u32, &[u8])],
    ) -> Result<()> {
        ensure!(
            !pages.is_empty() && pages.len() <= 16 && states.len() == 42,
            "invalid staged KV upload geometry"
        );
        ensure!(
            pages
                .iter()
                .map(|&(page, _)| page)
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                == pages.len(),
            "duplicate staged KV upload destination"
        );
        let maximum = pages
            .iter()
            .map(|&(page, _)| page as usize)
            .max()
            .unwrap_or(0);
        let host = self.host.as_mut_slice();
        for (layer, state) in states.iter().enumerate() {
            let State::Attention { k, v } = state else {
                anyhow::bail!("invalid KV upload state");
            };
            for (plane, buffer) in [k, v].iter().enumerate() {
                ensure!(
                    buffer.len() >= (maximum + 1) * 8192,
                    "KV upload plane is unavailable"
                );
                let at = (layer * 2 + plane) * 8;
                host[at..at + 8].copy_from_slice(&buffer.cu_deviceptr().to_le_bytes());
            }
        }
        for (index, &(page, payload)) in pages.iter().enumerate() {
            ensure!(payload.len() == PAGE, "invalid staged KV page extent");
            let at = (84 + index) * 8;
            host[at..at + 8].copy_from_slice(&u64::from(page).to_le_bytes());
            let at = HEADER + index * PAGE;
            host[at..at + PAGE].copy_from_slice(payload);
        }
        let bytes = HEADER + pages.len() * PAGE;
        let result = (|| -> Result<()> {
            // SAFETY: initialized pinned bytes and reserved device staging stay
            // live through the synchronization below, including enqueue failures.
            let status = unsafe {
                cuda_core::sys::cuMemcpyHtoDAsync_v2(
                    self.device.cu_deviceptr(),
                    self.host.as_ptr().cast(),
                    bytes,
                    stream.cu_stream(),
                )
            };
            ensure!(
                status == 0,
                "uploading staged KV bytes: {:?}",
                cuda_core::DriverError(status)
            );
            // SAFETY: complete aligned vectors, validated unique physical pages,
            // current plane pointers, and full page payloads behind the header.
            unsafe {
                module.llama_transfer.llama_upload_pages(
                    stream,
                    LaunchConfig::for_num_elems((pages.len() * PAGE / 16) as u32),
                    pages.len(),
                    &self.device,
                )?;
            }
            Ok(())
        })();
        let synchronized = stream.synchronize();
        result?;
        synchronized.context("finishing staged KV upload")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires CUDA"]
    fn pinned_upload_preserves_every_plane_and_noncontiguous_page() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.new_stream()?;
        let module = kernels::load(&context)?;
        let states: Vec<_> = (0..42)
            .map(|_| {
                Ok(State::Attention {
                    k: DeviceBuffer::zeroed(&stream, 3 * 8192)?,
                    v: DeviceBuffer::zeroed(&stream, 3 * 8192)?,
                })
            })
            .collect::<Result<_>>()?;
        let a: Vec<_> = (0..PAGE).map(|index| (index * 17 + 31) as u8).collect();
        let b: Vec<_> = (0..PAGE).map(|index| (index * 29 + 71) as u8).collect();
        let mut upload = Upload::new(&context, &stream)?;
        upload.write(&module, &stream, &states, &[(2, &a), (0, &b)])?;
        for (layer, state) in states.iter().enumerate() {
            let State::Attention { k, v } = state else {
                unreachable!()
            };
            for (plane, buffer) in [k, v].iter().enumerate() {
                let actual = buffer.to_host_vec(&stream)?;
                ensure!(
                    actual[8192..16384].iter().all(|&value| value == 0),
                    "untargeted KV page changed"
                );
                for (physical, expected) in [(0, &b), (2, &a)] {
                    let base = (layer * 2 + plane) * 16384;
                    for column in 0..8192 {
                        let at = base + column * 2;
                        let expected = u16::from_le_bytes([expected[at], expected[at + 1]]);
                        ensure!(
                            actual[physical * 8192 + column] == expected,
                            "KV upload changed layer{layer} plane{plane} page{physical} column{column}"
                        );
                    }
                }
            }
        }
        println!(
            "Pinned KV upload preserved all 42 layers, 2 planes, 2 distinct pages; middle page untouched"
        );
        Ok(())
    }
}
