//! CUDA Rust operator namespaces sharing one loaded device artifact.
use cuda_core::CudaContext;

mod attention;
mod awq;
mod awq_fp16;
mod awq_gateup;
mod awq_matrix;
mod awq_medium;
mod awq_split;
mod awq_tile;
mod basic;
#[cfg(test)]
mod bf16_cache;
mod bf16_pair;
mod bf16_router;
mod cfm_attention;
mod cfm_math;
mod codec;
mod codec_stream;
mod conv_parallel;
mod convolution;
#[cfg(test)]
mod delta;
mod delta128;
mod delta_chunk;
mod delta_mma;
mod delta_prepare;
mod delta_tiles;
mod dense_attention;
mod dense_simt;
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
mod float_matrix;
#[cfg(test)]
pub(super) mod fp8;
mod fp8_matmul;
mod fp8_prepare;
mod fp8_weights;
mod gsq;
mod gsq_tensor;
mod hift;
mod hift_decode;
mod hift_source;
mod hift_spectral;
pub(super) mod integer;
mod llama;
mod llama_bf16;
mod llama_flash;
mod llama_flash_kernel;
mod llama_flash_math;
mod llama_flash_register;
mod llama_fusion;
mod llama_gated;
mod llama_merge;
mod llama_nucleus;
mod llama_position;
mod llama_transfer;
pub(crate) mod matrix;
mod minicpm_bf16;
mod minicpm_vision;
mod moe;
mod moe_grouping;
mod moe_half;
#[cfg(test)]
mod moe_register;
mod moe_register_math;
mod moe_repack;
mod moe_residual;
mod moe_route;
mod moe_selected;
mod moe_tensor;
mod normalization;
mod packed_prefill;
mod penalty;
mod position_table;
mod qwen3_flash;
mod qwen3_prepare;
mod resampler;
mod rms_vector;
mod rotary;
mod router_fused;
mod s3;
mod selection;
mod speaker_math;
mod speaker_spatial;
mod speech;
mod tensor;
mod topk;
mod vision;
mod vision_attention;
mod vision_flash;
mod vision_math;
mod vision_position;
mod whisper;
mod wide_prefill;

pub struct LoadedModule {
    pub whisper: whisper::device::LoadedModule,
    pub speech: speech::device::LoadedModule,
    pub float_matrix: float_matrix::device::LoadedModule,
    pub codec_stream: codec_stream::device::LoadedModule,
    pub codec: codec::device::LoadedModule,
    pub hift_decode: hift_decode::device::LoadedModule,
    pub hift_source: hift_source::device::LoadedModule,
    pub hift_spectral: hift_spectral::device::LoadedModule,
    pub s3: s3::device::LoadedModule,
    pub speaker_math: speaker_math::device::LoadedModule,
    pub speaker_spatial: speaker_spatial::device::LoadedModule,
    pub dense_simt: dense_simt::device::LoadedModule,
    pub hift: hift::device::LoadedModule,
    pub cfm_math: cfm_math::device::LoadedModule,
    pub cfm_attention: cfm_attention::device::LoadedModule,
    pub resampler: resampler::device::LoadedModule,
    pub awq: awq::device::LoadedModule,
    pub awq_gateup: awq_gateup::device::LoadedModule,
    pub awq_fp16: awq_fp16::device::LoadedModule,
    pub awq_tile: awq_tile::device::LoadedModule,
    pub awq_split: awq_split::device::LoadedModule,
    pub awq_medium: awq_medium::device::LoadedModule,
    pub awq_matrix: awq_matrix::device::LoadedModule,
    pub dense_attention: dense_attention::device::LoadedModule,
    pub llama: llama::device::LoadedModule,
    pub llama_fusion: llama_fusion::device::LoadedModule,
    pub llama_gated: llama_gated::device::LoadedModule,
    pub llama_merge: llama_merge::device::LoadedModule,
    pub llama_position: llama_position::device::LoadedModule,
    pub llama_transfer: llama_transfer::device::LoadedModule,
    pub llama_bf16: llama_bf16::device::LoadedModule,
    pub llama_nucleus: llama_nucleus::device::LoadedModule,
    pub llama_flash: llama_flash::device::LoadedModule,
    pub qwen3_prepare: qwen3_prepare::device::LoadedModule,
    pub qwen3_flash: qwen3_flash::device::LoadedModule,
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
    pub minicpm_vision: minicpm_vision::device::LoadedModule,
    pub minicpm_bf16: minicpm_bf16::device::LoadedModule,
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
    pub rms_vector: rms_vector::device::LoadedModule,
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
    // Allow attention CTAs to use the device's available shared-memory capacity.
    for name in [
        "llama_flash128",
        "llama_flash_wide",
        "llama_flash_decode",
        "qwen3_flash",
    ] {
        let function = module.load_function(name)?;
        let status = unsafe {
            cuda_core::sys::cuFuncSetAttribute(
                function.cu_function(),
                cuda_core::sys::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                100,
            )
        };
        anyhow::ensure!(
            status == 0,
            "set {name} shared-memory preference: {status:?}"
        );
    }

    Ok(LoadedModule {
        resampler: resampler::device::from_module(module.clone())?,
        hift_decode: hift_decode::device::from_module(module.clone())?,
        hift_source: hift_source::device::from_module(module.clone())?,
        hift_spectral: hift_spectral::device::from_module(module.clone())?,
        s3: s3::device::from_module(module.clone())?,
        speaker_math: speaker_math::device::from_module(module.clone())?,
        speaker_spatial: speaker_spatial::device::from_module(module.clone())?,
        dense_simt: dense_simt::device::from_module(module.clone())?,
        hift: hift::device::from_module(module.clone())?,
        cfm_math: cfm_math::device::from_module(module.clone())?,
        cfm_attention: cfm_attention::device::from_module(module.clone())?,
        codec: codec::device::from_module(module.clone())?,
        codec_stream: codec_stream::device::from_module(module.clone())?,
        float_matrix: float_matrix::device::from_module(module.clone())?,
        speech: speech::device::from_module(module.clone())?,
        whisper: whisper::device::from_module(module.clone())?,
        awq: awq::device::from_module(module.clone())?,
        awq_gateup: awq_gateup::device::from_module(module.clone())?,
        awq_fp16: awq_fp16::device::from_module(module.clone())?,
        awq_tile: awq_tile::device::from_module(module.clone())?,
        awq_split: awq_split::device::from_module(module.clone())?,
        awq_medium: awq_medium::device::from_module(module.clone())?,
        awq_matrix: awq_matrix::device::from_module(module.clone())?,
        dense_attention: dense_attention::device::from_module(module.clone())?,
        llama: llama::device::from_module(module.clone())?,
        llama_fusion: llama_fusion::device::from_module(module.clone())?,
        llama_gated: llama_gated::device::from_module(module.clone())?,
        llama_merge: llama_merge::device::from_module(module.clone())?,
        llama_position: llama_position::device::from_module(module.clone())?,
        llama_transfer: llama_transfer::device::from_module(module.clone())?,
        llama_bf16: llama_bf16::device::from_module(module.clone())?,
        llama_nucleus: llama_nucleus::device::from_module(module.clone())?,
        llama_flash: llama_flash::device::from_module(module.clone())?,
        qwen3_prepare: qwen3_prepare::device::from_module(module.clone())?,
        qwen3_flash: qwen3_flash::device::from_module(module.clone())?,
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
        minicpm_vision: minicpm_vision::device::from_module(module.clone())?,
        minicpm_bf16: minicpm_bf16::device::from_module(module.clone())?,
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
        rms_vector: rms_vector::device::from_module(module.clone())?,
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
