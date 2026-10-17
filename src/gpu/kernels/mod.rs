//! CUDA Rust operator namespaces sharing one loaded device artifact.
use cuda_core::CudaContext;

mod attention;
mod basic;
mod convolution;
#[cfg(test)]
mod delta;
mod delta128;
mod delta_chunk;
mod delta_mma;
mod delta_tiles;
#[cfg(test)]
mod flash;
mod flash_half;
mod flash_mma;
pub(super) mod integer;
pub(crate) mod matrix;
mod normalization;
mod packed_prefill;
mod rotary;
mod selection;
mod tensor;
mod topk;
mod wide_prefill;

pub struct LoadedModule {
    pub packed_prefill: packed_prefill::device::LoadedModule,
    pub tensor: tensor::device::LoadedModule,
    pub wide_prefill: wide_prefill::device::LoadedModule,
    pub basic: basic::device::LoadedModule,
    pub normalization: normalization::device::LoadedModule,
    pub selection: selection::device::LoadedModule,
    pub topk: topk::device::LoadedModule,
    pub convolution: convolution::device::LoadedModule,
    #[cfg(test)]
    pub delta: delta::device::LoadedModule,
    pub delta128: delta128::device::LoadedModule,
    pub delta_chunk: delta_chunk::device::LoadedModule,
    pub rotary: rotary::device::LoadedModule,
    #[cfg(test)]
    pub flash: flash::device::LoadedModule,
    pub flash_half: flash_half::device::LoadedModule,
    pub flash_mma: flash_mma::device::LoadedModule,
    pub attention: attention::device::LoadedModule,
    pub integer: integer::device::LoadedModule,
    pub matrix: matrix::device::LoadedModule,
}

pub fn load(context: &std::sync::Arc<CudaContext>) -> anyhow::Result<LoadedModule> {
    // The concrete module loader retains the library's artifact through its
    // linker anchor. All namespaces share that single loaded bundle.
    let basic = basic::device::load(context)?;
    let module = basic.as_cuda_module().clone();
    Ok(LoadedModule {
        packed_prefill: packed_prefill::device::from_module(module.clone())?,
        tensor: tensor::device::from_module(module.clone())?,
        wide_prefill: wide_prefill::device::from_module(module.clone())?,
        basic,
        normalization: normalization::device::from_module(module.clone())?,
        selection: selection::device::from_module(module.clone())?,
        topk: topk::device::from_module(module.clone())?,
        convolution: convolution::device::from_module(module.clone())?,
        #[cfg(test)]
        delta: delta::device::from_module(module.clone())?,
        delta128: delta128::device::from_module(module.clone())?,
        delta_chunk: delta_chunk::device::from_module(module.clone())?,
        rotary: rotary::device::from_module(module.clone())?,
        #[cfg(test)]
        flash: flash::device::from_module(module.clone())?,
        flash_half: flash_half::device::from_module(module.clone())?,
        flash_mma: flash_mma::device::from_module(module.clone())?,
        attention: attention::device::from_module(module.clone())?,
        integer: integer::device::from_module(module.clone())?,
        matrix: matrix::device::from_module(module.clone())?,
    })
}
