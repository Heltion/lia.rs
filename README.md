# lia.rs

**L**ocal **I**nference for **A**gent in **R**u**s**t

A local inference engine for coding agents, independently implemented in Rust
with [NVIDIA CUDA Rust](https://github.com/NVIDIA/cuda-rust). It runs supported
Qwen and MiniCPM checkpoints through a Chat Completions API, with image input,
thinking, tool calls, streaming, and native multi-token prediction (MTP) where
model weights support it.

MiniCPM5-2B supports text generation, optional request batching, and prefix
caching with host-memory and SSD storage. Both cache tiers work with batching
enabled or disabled:

```sh
lia serve /path/to/MiniCPM5-2B 8080 32768 --batch --max-sequences 8 \
  --kv-pool-tokens 65536 --kv-host-mib 1024 --kv-ssd-mib 2048
```

Use `--no-batch` for serial execution, the default. Cache capacities and the cache
directory (`--kv-cache-dir`) are configurable independently of request batching.

This project is in early development. It currently supports only a small number
of models, and performance still needs improvement.
