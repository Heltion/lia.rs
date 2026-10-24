//! One contiguous upload scatters complete immutable KV pages across all planes.
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    struct Words {
        data: [u32; 4],
    }

    #[kernel]
    pub unsafe fn llama_upload_pages(count: usize, staged: &[u8]) {
        let index = thread::index_1d().get();
        const PAGE_VECTORS: usize = 42 * 2 * 32 * 256 * 2 / 16;
        if index < count * PAGE_VECTORS {
            let page = index / PAGE_VECTORS;
            let local = index % PAGE_VECTORS;
            let plane = local / 1024;
            let offset = local % 1024 * 16;
            unsafe {
                let metadata = staged.as_ptr().cast::<u64>();
                let base = *metadata.add(plane);
                let physical = *metadata.add(84 + page);
                let source = staged.as_ptr().add(1024 + index * 16).cast::<Words>();
                let destination = (base + physical * 16384 + offset as u64) as *mut Words;
                destination.write(source.read());
            }
        }
    }
}
