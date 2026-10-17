# lia.rs

Local Inference for Agent in Rust. MIT licensed; developed independently from
an empty main branch. It implements a single-sequence Qwen 3.5 architecture
(`qwen35` GGUF metadata), including its hybrid recurrent/attention trunk and
native MTP block.

The target is GGUF Qwen 3.5 hybrid attention/recurrent inference with thinking
and native MTP, implemented in Rust using NVIDIA CUDA Rust. llama.cpp and the
official Gorilla BFCL v3 evaluator are external validation references only.

## Build and inspect

```text
cargo run --release -- inspect /path/to/model.gguf
cargo run --release -- tokenize /path/to/model.gguf 'Hello, 世界'
cargo test
```

## CUDA kernels

Install the pinned Rust toolchain and NVIDIA CUDA Rust's `cargo-oxide`, and
configure a CUDA 13 toolkit as described in its official documentation:
https://github.com/NVIDIA/cuda-rust

```text
cargo oxide build --arch sm_120 -- --release --features cuda
target/release/lia check-gpu
target/release/lia check-prefill-gpu
target/release/lia check-model-gpu /path/to/model.gguf
target/release/lia forward /path/to/model.gguf 109266,3709
target/release/lia serve /path/to/model.gguf 18474 16384
```

Change the architecture for your GPU. The CUDA feature contains Rust kernels;
it does not delegate model inference to another engine.

Keep model weights, external source checkouts, BFCL fixtures, evaluation
outputs, logs, profiling data and temporary files outside this repository.
See `THIRD_PARTY_NOTICES` for quantization codebook attribution.

The local server provides `/v1/chat/completions`, `/apply-template`, `/tokenize`
and `/health`. It uses greedy sampling, thinking, configurable native MTP
proposals, and batched target verification. Rejected suffixes restore the
per-position recurrent state captured during verification. Chat requests are
serialized and use one conversation cache, with recurrent-state checkpoints
for prompt changes.
The model's own GGUF chat template supplies prompts and tool formatting.

Decode and small batches use signed integer dot products with per-group
activation quantization and paired output rows. Prefill uses batches of up to
512 tokens. Batches above eight use Rust Tensor Core kernels with f32
accumulation: short batches decode packed weights directly into shared tiles,
while larger batches unpack weights into a reusable f16 workspace. `LIA_INT8=0` selects the
floating-point activation reference path for decode and small batches. `LIA_MM_ROWS=1` selects unpaired
integer rows; `LIA_SPLIT_K=4` splits paired rows across four warps (it requires
`LIA_MM_ROWS=2`, the default). `LIA_PROFILE=1` adds synchronized matrix timing
diagnostics; disable it for end-to-end measurements.

`LIA_MTP_DRAFT_MAX` accepts 1–7 and tunes native MTP chain length (default 3).
Verification always checks drafts against the target; this setting does not
disable thinking or native MTP. KV caches use f16 and accumulate in f32.
