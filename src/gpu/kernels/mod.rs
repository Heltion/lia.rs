//! CUDA Rust operator namespaces sharing one loaded device artifact.
use cuda_core::CudaContext;

mod attention;
mod basic;
#[cfg(test)]
mod bf16_cache;
mod bf16_pair;
mod bf16_router;
mod conv_parallel;
mod convolution;
#[cfg(test)]
mod delta;
mod delta128;
mod delta_chunk;
mod delta_mma;
mod delta_prepare;
mod delta_tiles;
#[cfg(test)]
mod flash;
#[cfg(test)]
mod flash_bf16;
#[cfg(test)]
mod flash_bf16_math;
#[cfg(test)]
pub(crate) mod flash_compact;
#[cfg(test)]
mod flash_compact_pv;
mod flash_half;
mod flash_merge;
mod flash_mma;
mod flash_mma64;
mod flash_mma_math;
#[cfg(test)]
pub(super) mod fp8;
mod fp8_matmul;
mod fp8_prepare;
mod fp8_weights;
mod gsq;
mod gsq_tensor;
pub(super) mod integer;
pub(crate) mod matrix;
mod moe;
mod moe_grouping;
mod moe_half;
#[cfg(test)]
mod moe_register;
mod moe_register_math;
mod moe_repack;
mod moe_route;
mod moe_selected;
mod moe_tensor;
mod normalization;
mod packed_prefill;
mod penalty;
mod position_table;
mod rotary;
mod router_fused;
mod selection;
mod tensor;
mod topk;
mod vision;
mod vision_attention;
mod vision_flash;
mod vision_math;
mod vision_position;
mod wide_prefill;

pub struct LoadedModule {
    pub bf16_pair: bf16_pair::device::LoadedModule,
    pub router_fused: router_fused::device::LoadedModule,
    pub bf16_router: bf16_router::device::LoadedModule,
    pub gsq: gsq::device::LoadedModule,
    pub gsq_tensor: gsq_tensor::device::LoadedModule,
    pub moe: moe::device::LoadedModule,
    #[cfg(test)]
    pub moe_half: moe_half::device::LoadedModule,
    pub moe_grouping: moe_grouping::device::LoadedModule,
    pub moe_selected: moe_selected::device::LoadedModule,
    #[cfg(test)]
    pub moe_register: moe_register::device::LoadedModule,
    pub moe_repack: moe_repack::device::LoadedModule,
    pub moe_tensor: moe_tensor::device::LoadedModule,
    pub penalty: penalty::device::LoadedModule,
    pub position_table: position_table::device::LoadedModule,
    pub vision: vision::device::LoadedModule,
    pub vision_math: vision_math::device::LoadedModule,
    pub vision_flash: vision_flash::device::LoadedModule,
    pub vision_attention: vision_attention::device::LoadedModule,
    pub vision_position: vision_position::device::LoadedModule,
    #[cfg(test)]
    pub fp8: fp8::device::LoadedModule,
    pub fp8_weights: fp8_weights::device::LoadedModule,
    pub fp8_prepare: fp8_prepare::device::LoadedModule,
    pub fp8_matmul: fp8_matmul::device::LoadedModule,
    pub packed_prefill: packed_prefill::device::LoadedModule,
    pub tensor: tensor::device::LoadedModule,
    pub wide_prefill: wide_prefill::device::LoadedModule,
    pub basic: basic::device::LoadedModule,
    pub normalization: normalization::device::LoadedModule,
    pub selection: selection::device::LoadedModule,
    pub topk: topk::device::LoadedModule,
    pub conv_parallel: conv_parallel::device::LoadedModule,
    pub convolution: convolution::device::LoadedModule,
    #[cfg(test)]
    pub delta: delta::device::LoadedModule,
    pub delta128: delta128::device::LoadedModule,
    pub delta_chunk: delta_chunk::device::LoadedModule,
    pub delta_prepare: delta_prepare::device::LoadedModule,
    pub rotary: rotary::device::LoadedModule,
    #[cfg(test)]
    pub flash: flash::device::LoadedModule,
    pub flash_half: flash_half::device::LoadedModule,
    pub flash_merge: flash_merge::device::LoadedModule,
    #[cfg(test)]
    pub flash_compact: flash_compact::device::LoadedModule,
    #[cfg(test)]
    pub flash_compact_pv: flash_compact_pv::device::LoadedModule,
    pub flash_mma64: flash_mma64::device::LoadedModule,
    #[cfg(test)]
    pub flash_bf16: flash_bf16::device::LoadedModule,
    #[cfg(test)]
    pub bf16_cache: bf16_cache::device::LoadedModule,
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
        bf16_pair: bf16_pair::device::from_module(module.clone())?,
        // SAFETY: bundled fused-router contract requires exactly256threads.
        router_fused: unsafe { router_fused::device::from_module(module.clone())? },
        bf16_router: bf16_router::device::from_module(module.clone())?,
        gsq: gsq::device::from_module(module.clone())?,
        gsq_tensor: gsq_tensor::device::from_module(module.clone())?,
        moe: moe::device::from_module(module.clone())?,
        #[cfg(test)]
        moe_half: moe_half::device::from_module(module.clone())?,
        moe_grouping: moe_grouping::device::from_module(module.clone())?,
        moe_selected: moe_selected::device::from_module(module.clone())?,
        #[cfg(test)]
        moe_register: moe_register::device::from_module(module.clone())?,
        moe_repack: moe_repack::device::from_module(module.clone())?,
        moe_tensor: moe_tensor::device::from_module(module.clone())?,
        penalty: penalty::device::from_module(module.clone())?,
        position_table: position_table::device::from_module(module.clone())?,
        vision: vision::device::from_module(module.clone())?,
        vision_math: vision_math::device::from_module(module.clone())?,
        vision_flash: vision_flash::device::from_module(module.clone())?,
        vision_attention: vision_attention::device::from_module(module.clone())?,
        vision_position: vision_position::device::from_module(module.clone())?,
        #[cfg(test)]
        fp8: fp8::device::from_module(module.clone())?,
        fp8_weights: fp8_weights::device::from_module(module.clone())?,
        fp8_prepare: fp8_prepare::device::from_module(module.clone())?,
        fp8_matmul: fp8_matmul::device::from_module(module.clone())?,
        packed_prefill: packed_prefill::device::from_module(module.clone())?,
        tensor: tensor::device::from_module(module.clone())?,
        wide_prefill: wide_prefill::device::from_module(module.clone())?,
        basic,
        normalization: normalization::device::from_module(module.clone())?,
        selection: selection::device::from_module(module.clone())?,
        topk: topk::device::from_module(module.clone())?,
        conv_parallel: conv_parallel::device::from_module(module.clone())?,
        convolution: convolution::device::from_module(module.clone())?,
        #[cfg(test)]
        delta: delta::device::from_module(module.clone())?,
        delta128: delta128::device::from_module(module.clone())?,
        delta_chunk: delta_chunk::device::from_module(module.clone())?,
        delta_prepare: delta_prepare::device::from_module(module.clone())?,
        rotary: rotary::device::from_module(module.clone())?,
        #[cfg(test)]
        flash: flash::device::from_module(module.clone())?,
        flash_half: flash_half::device::from_module(module.clone())?,
        flash_merge: flash_merge::device::from_module(module.clone())?,
        // SAFETY: the shared artifact contains this exact kernel's launch
        // contracts; the module was loaded through its bundled linker anchor.
        // SAFETY: generated contract validates 1024 threads and aligned dynamic shared storage.
        flash_mma64: unsafe { flash_mma64::device::from_module(module.clone())? },
        // SAFETY: test launch checks the bundled 99-KiB dynamic-shared contract.
        #[cfg(test)]
        flash_compact: unsafe { flash_compact::device::from_module(module.clone())? },
        // SAFETY: test launch validates the same bundled dynamic-shared contract.
        #[cfg(test)]
        flash_compact_pv: unsafe { flash_compact_pv::device::from_module(module.clone())? },
        #[cfg(test)]
        flash_bf16: unsafe { flash_bf16::device::from_module(module.clone())? },
        #[cfg(test)]
        bf16_cache: bf16_cache::device::from_module(module.clone())?,
        flash_mma: unsafe { flash_mma::device::from_module(module.clone())? },
        attention: attention::device::from_module(module.clone())?,
        integer: integer::device::from_module(module.clone())?,
        matrix: matrix::device::from_module(module.clone())?,
    })
}
