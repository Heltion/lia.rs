# lia.rs

**L**ocal **I**nference for **A**gent in **R**u**s**t

A local inference engine for coding agents, implemented independently in Rust
with [NVIDIA CUDA Rust](https://github.com/NVIDIA/cuda-rust). It serves local
GGUF models through a Chat Completions API, with thinking, tool calls, streaming,
and native multi-token prediction (MTP).

The examples below show Pi, DeepSeek Harness, and OpenCode using lia.rs to turn
the same prompt into a self-contained animated SVG page.

## A pelican riding a bicycle

> Create a self-contained HTML file with a 2D SVG animation of a pelican riding a bicycle

**Model:** `Qwen3.8-27B-GSQ-RCO-IQ2_XS-mtp.gguf`

| Environment | Configuration |
| --- | --- |
| OS | Ubuntu 24.04 LTS |
| GPU / VRAM | NVIDIA GeForce RTX 5060 Ti / 16 GB |
| System RAM | 32 GB |
| CUDA | 13 |

All three native agents used **xhigh thinking**, a 49,152-token context window,
and native MTP with a draft length of 1. The previews were captured from the
final generated HTML without manual repairs.

| Agent | Generated animation | Task time |
| --- | :---: | ---: |
| **[Pi](docs/pelican/README.md#pi)**<br>1.3.0<br>Thinking: xhigh | <img src="docs/pelican/pi.gif" alt="Pi-generated pelican riding a bicycle" width="480"> | **637.01 s** |
| **[DeepSeek Harness](docs/pelican/README.md#deepseek-harness)**<br>0.2.0-rc.2<br>Thinking: xhigh | <img src="docs/pelican/deepseek-harness.gif" alt="DeepSeek Harness-generated bird riding a bicycle" width="480"> | **725.88 s** |
| **[OpenCode](docs/pelican/README.md#opencode)**<br>2.0.32<br>Thinking: xhigh | <img src="docs/pelican/opencode.gif" alt="OpenCode-generated pelican riding a bicycle" width="480"> | **655.29 s** |

Task time covers the complete agent run after the inference server is ready,
including model requests, tools, and native auxiliary work. These are single
runs with the same thinking and sampling settings; agent prompts, tools, and
generated trajectories still differ. All animations run, but pelican anatomy
is approximate; visual quality is a qualitative check.

See [the pelican test](docs/pelican/README.md) for the model, configuration,
testing procedure, and timing definitions.
