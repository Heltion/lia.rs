# lia.rs

**L**ocal **I**nference for **A**gent in **R**u**s**t

A local inference engine for coding agents, independently implemented in Rust
with [NVIDIA CUDA Rust](https://github.com/NVIDIA/cuda-rust). It runs supported
Qwen GGUF and GSQ checkpoints through a Chat Completions API, with image input,
thinking, tool calls, streaming, and native multi-token prediction (MTP) where
model weights support it.

This project is in early development. It currently supports only a small number
of models, and performance still needs improvement.
