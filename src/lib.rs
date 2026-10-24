#![feature(f16)]
//! Independent Rust model loading, text processing, and CUDA inference.
pub mod chat;
mod codebooks;
pub mod gguf;
#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(feature = "image-tools")]
pub mod image_tools;
#[cfg(feature = "cuda")]
pub mod inference;
pub mod models;
pub mod onnx_weights;
mod prefix_cache;
pub mod quant;
pub mod safetensors;
pub mod sampling;
#[cfg(feature = "cuda")]
pub mod server;
pub mod tokenizer;
pub mod torch_weights;
pub mod vision;

/// Validated index bound; model metadata may declare a smaller context.
pub(crate) const MAX_CONTEXT_TOKENS: usize = 262144;
