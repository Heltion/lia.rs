//! CUDA Rust operator namespaces sharing one loaded device artifact.
use cuda_core::CudaContext;

mod attention;
mod basic;
mod convolution;
mod delta;
mod delta128;
mod flash;
mod flash_half;
pub(super) mod integer;
pub(crate) mod matrix;
mod packed_prefill;
mod rotary;
mod tensor;

pub struct LoadedModule {
    pub packed_prefill: packed_prefill::device::LoadedModule,
    pub tensor: tensor::device::LoadedModule,
    pub basic: basic::device::LoadedModule,
    pub convolution: convolution::device::LoadedModule,
    pub delta: delta::device::LoadedModule,
    pub delta128: delta128::device::LoadedModule,
    pub rotary: rotary::device::LoadedModule,
    pub flash: flash::device::LoadedModule,
    pub flash_half: flash_half::device::LoadedModule,
    pub attention: attention::device::LoadedModule,
    pub integer: integer::device::LoadedModule,
    pub matrix: matrix::device::LoadedModule,
}

pub fn load(context: &std::sync::Arc<CudaContext>) -> anyhow::Result<LoadedModule> {
    // Every kernel is emitted into this crate's one embedded artifact.
    // Load it directly rather than parsing and merging the same PTX bundle.
    let module = cuda_host::load_embedded_module(context, env!("CARGO_PKG_NAME"))?;
    Ok(LoadedModule {
        packed_prefill: packed_prefill::device::from_module(module.clone())?,
        tensor: tensor::device::from_module(module.clone())?,
        basic: basic::device::from_module(module.clone())?,
        convolution: convolution::device::from_module(module.clone())?,
        delta: delta::device::from_module(module.clone())?,
        delta128: delta128::device::from_module(module.clone())?,
        rotary: rotary::device::from_module(module.clone())?,
        flash: flash::device::from_module(module.clone())?,
        flash_half: flash_half::device::from_module(module.clone())?,
        attention: attention::device::from_module(module.clone())?,
        integer: integer::device::from_module(module.clone())?,
        matrix: matrix::device::from_module(module.clone())?,
    })
}
