//! Four warps own sixteen GQA rows each, including their complete score reductions.
use cuda_device::{kernel, launch_bounds};
use cuda_host::cuda_module;
#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::super::llama_flash_kernel::attention;
    use super::*;

    #[kernel]
    #[launch_bounds(128, 3)]
    pub unsafe fn llama_flash128(
        query_start: usize,
        query_count: usize,
        position: usize,
        parts: usize,
        span: usize,
        paged: bool,
        stride: usize,
        positions: &[u32],
        sequences: &[u32],
        pages: &[u32],
        q: &[f32],
        k: &[u16],
        v: &[u16],
        normalize: bool,
        partial: *mut f32,
    ) {
        unsafe {
            attention::<64, 128, 32, 2, 8, false, false, false>(
                query_start,
                query_count,
                position,
                parts,
                span,
                paged,
                stride,
                positions,
                sequences,
                pages,
                q,
                k,
                v,
                normalize,
                partial,
            );
        }
    }
    #[kernel]
    #[launch_bounds(128, 2)]
    pub unsafe fn llama_flash_wide(
        query_start: usize,
        query_count: usize,
        position: usize,
        parts: usize,
        span: usize,
        paged: bool,
        stride: usize,
        positions: &[u32],
        sequences: &[u32],
        pages: &[u32],
        q: &[f32],
        k: &[u16],
        v: &[u16],
        normalize: bool,
        partial: *mut f32,
    ) {
        unsafe {
            attention::<64, 128, 64, 2, 8, false, false, false>(
                query_start,
                query_count,
                position,
                parts,
                span,
                paged,
                stride,
                positions,
                sequences,
                pages,
                q,
                k,
                v,
                normalize,
                partial,
            );
        }
    }
    #[kernel]
    #[launch_bounds(32, 6)]
    pub unsafe fn llama_flash_decode(
        query_start: usize,
        query_count: usize,
        position: usize,
        parts: usize,
        span: usize,
        paged: bool,
        stride: usize,
        positions: &[u32],
        sequences: &[u32],
        pages: &[u32],
        q: &[f32],
        k: &[u16],
        v: &[u16],
        normalize: bool,
        partial: *mut f32,
    ) {
        unsafe {
            attention::<16, 32, 32, 2, 8, false, false, false>(
                query_start,
                query_count,
                position,
                parts,
                span,
                paged,
                stride,
                positions,
                sequences,
                pages,
                q,
                k,
                v,
                normalize,
                partial,
            );
        }
    }
}
