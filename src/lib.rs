#![feature(f16)]
//! Independent Rust model loading, text processing, and CUDA inference.
pub mod chat;
mod codebooks;
pub mod gguf;
#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(feature = "cuda")]
pub mod inference;
pub mod quant;
pub mod sampling;
#[cfg(feature = "cuda")]
pub mod server;
pub mod tokenizer;
pub mod vision;
