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
target/release/lia forward /path/to/model.gguf 109266,3709
target/release/lia serve /path/to/model.gguf 18474 16384
```

Change the architecture for your GPU. The CUDA feature contains Rust kernels;
it does not delegate model inference to another engine.

GPU operator unit tests are compiled only for tests. Run them serially on a
CUDA device; the ignored model tests additionally require an external model:

```text
cargo oxide test --arch sm_120 -- --release --features cuda --lib -- --test-threads=1
LIA_TEST_MODEL=/path/to/model.gguf cargo oxide test --arch sm_120 -- --release --features cuda -- --ignored --test-threads=1
```

Keep model weights, external source checkouts, BFCL fixtures, evaluation
outputs, logs, profiling data and temporary files outside this repository.
See `THIRD_PARTY_NOTICES` for quantization codebook attribution.

The local server provides `/v1/chat/completions`, `/apply-template`, `/tokenize`
and `/health`. It supports greedy or seeded top-k/top-p sampling, thinking, configurable native MTP
proposals, and batched target verification. Rejected suffixes restore the
per-position recurrent state captured during verification. Chat requests are
serialized and use one conversation cache, with recurrent-state checkpoints
for prompt changes.
The model's own GGUF chat template supplies prompts and tool formatting.

Decode and small batches use signed integer dot products with per-group
activation quantization and paired output rows. Prefill uses batches of up to
512 tokens. Batches above eight use Rust Tensor Core kernels with f32
accumulation. Packed weights are decoded into shared tiles reused across
32 token columns below batch 33, 64 columns for batches 33–64, and 128 columns
from batch 65. `LIA_INT8=0` selects the
floating-point activation reference path for decode and small batches.
Projection groups share activation preparation for one immutable input, then
prepare it again for the next group.
`LIA_MM_ROWS=1` selects unpaired integer rows; `LIA_SPLIT_K=4` splits paired
rows across four warps (it requires
`LIA_MM_ROWS=2`, the default). `LIA_MM_THREADS` selects 64, 128 (default), or
256 threads for unsplit integer matrix blocks; split blocks use 128.
`LIA_PROFILE=1` adds synchronized matrix timing diagnostics; disable it for
end-to-end measurements.

`LIA_MTP_DRAFT_MAX` accepts 1–7 and tunes native MTP chain length (default 3).
Verification always checks drafts against the target; this setting does not
disable thinking or native MTP. KV caches use f16 and accumulate in f32.

Recurrent prefill solves 16-token triangular chunks with BF16 high/residual
Tensor Core products. Recurrent state and persistent accumulators remain f32.
MTP verification retains per-position state capture in the register-based path.

The CUDA library target uses release builds without LTO. Builds are serialized
to avoid intermediate-file collisions in the pinned CUDA backend.

Attention uses 32-bit indexing within the validated context and buffer extents.
Decode partitions contain 64 tokens. Prefill batches of at least 16 queries
share KV tiles across 16 queries using Tensor Core attention with 256-token
partitions. FP16 query and probability residuals preserve f32 inputs and
accumulation; smaller prefill batches use 128-token scalar partitions.
Large RMSNorm rows use cooperative 256-thread blocks with f32 reduction;
short rows retain warp reductions.

Greedy decoding remains the default (`temperature=0`). Positive temperatures
enable seeded sampling with `top_k` from 1 to 20 and `top_p` in (0, 1]; defaults
are `top_k=20`, `top_p=0.95`, and `seed=42`. Nondefault probability penalties are
not supported. Rust GPU kernels extract exact top-20 candidates per partition
before host probability normalization and sampling. Native MTP uses greedy
proposals and causal target sample-and-match verification; rejected suffixes
neither advance the target random stream nor retain recurrent state.
