# Pelican Test

Run each native agent against lia.rs with this exact task:

> Create a self-contained HTML file with a 2D SVG animation of a pelican riding a bicycle

## Results and timing

| Agent | Version | Thinking | Preparation | Task time | Requests |
| --- | --- | --- | ---: | ---: | ---: |
| Pi | 1.3.0 | xhigh | 4.62 s | 637.01 s | 4 |
| DeepSeek Harness | 0.2.0-rc.2 | xhigh | 4.62 s | 725.88 s | 11 |
| OpenCode | 2.0.32 | xhigh | 4.62 s | 655.29 s | 5 |

These runs were recorded on 2026-10-18. Task time is monotonic elapsed time from
launching the native CLI until it exits, after the server is ready. It includes
all model requests, tools, auxiliary work, and any native compaction. Preparation
is measured separately from server launch until its health endpoint is ready.
The single runs share thinking and sampling settings, but native system prompts,
tools, and generated trajectories differ; they do not establish a general
ranking of agent speed. No llama.cpp comparison was performed in this test.

All three clients completed their main tasks and ended normally. Every recorded
request used xhigh, and server timing records confirmed the shared sampling and
native MTP proposals and acceptance. No main-task request was truncated.
DeepSeek Harness's native title request hit its separate 64-token cap; its main
task continued normally. All 20 requests returned complete HTTP 200 streams.

Browser inspection of the final HTML found animated inline SVG, no external
resource requests, and no console errors. All three depict stylized birds on
bicycles; pelican anatomy and pedaling geometry are approximate.
These are qualitative observations without an official score. The six-second
GIFs use 72 distinct frames at 12 fps, resized to 720 pixels wide. Generated
HTML was not manually repaired.

A preliminary browser inspection briefly overlapped Pi's second prefill; its
reported task time is unadjusted. Final media for all three were captured after
all CLI runs finished. Raw attempts, failures, requests, timing records, and
HTML remain outside Git; only these selected previews are checked in.

### Pi

![Pi result](pi.gif)

### DeepSeek Harness

![DeepSeek Harness result](deepseek-harness.gif)

### OpenCode

![OpenCode result](opencode.gif)

## Test setup

The recorded runs used Ubuntu 24.04 LTS, CUDA 13, an RTX 5060 Ti with 16 GB
VRAM, and 32 GB system RAM. The model was
`Qwen3.8-27B-GSQ-RCO-IQ2_XS-mtp.gguf` with SHA-256
`f3369f8d36968ab7dd12f5ff11452680aef9371aa19c9922d22ccd5e58c2d9c8`.
The lia.rs binary SHA-256 was
`c59e3474a49bdb1ce1d2c6de42d35d85229285d7cbb30ad368539a5650da3116`.

Use a compatible CUDA environment and the repository's pinned Rust toolchain.
Install `cargo-oxide` following [NVIDIA CUDA Rust](https://github.com/NVIDIA/cuda-rust).
Build for the GPU architecture; these runs used:

```sh
cargo oxide build --arch sm_120 -- --release --features cuda
```

Keep model weights, agent installations, configurations, workspaces, sessions,
and raw records outside the repository. Use a Node.js version supported by all
three pinned agent packages; the recorded runs used Node.js 24.22.0. Set
`TEST_ROOT` to an absolute directory outside the repository and `MODEL` to the
GGUF path. Install the native clients in that external directory:

```sh
npm install --prefix "$TEST_ROOT/runtime" \
  @earendil-works/pi-coding-agent@1.3.0 \
  @deepseek-ai/dsh@0.2.0-rc.2 @opencode/cli@2.0.32
export PATH="$TEST_ROOT/runtime/node_modules/.bin:$PATH"
```

Start a fresh inference service for each agent:

```sh
LIA_CUDA_GRAPH=1 LIA_PREALLOCATE=1 \
LIA_ATTN_MMA_DECODE=1 LIA_ATTN_QUERY_TILES=1 LIA_GDN_PARALLEL=0 \
LIA_PREFILL_TILE=128 LIA_PREFILL_BATCH=128 \
LIA_MTP_SAMPLING=stochastic LIA_MTP_DRAFT_MAX=1 \
LIA_PREFILL_FP8=1 LIA_PREFILL_WEIGHT_PREPACK=0 \
LIA_MM_ROWS=2 LIA_SPLIT_K=1 LIA_MM_THREADS=128 LIA_INT8=1 \
  target/release/lia serve "$MODEL" 18473 49152
```

Use the same binary and environment for every run. The 49,152-token cache and
MTP replay through its final position were validated with the real model before
freezing this configuration. Larger 65,536-token probes ran out of VRAM on this
machine; 49,152 tokens with draft length 3 also lacked buffer headroom.

Wait for `http://127.0.0.1:18473/health` to report ready. Record its response and
verify thinking and native MTP in actual responses, including draft proposals
and acceptance. Sampling is temperature 1, top-k 20, top-p 0.95, and seed 42.
Pi and OpenCode set these explicitly below; DeepSeek Harness uses these server
defaults. Quantization settings are identical across clients.
The recorded runs used an external pass-through recorder on port 18475, which
forwarded requests unchanged to 18473; the examples below connect directly.

Create an empty workspace for each agent. Set `PWD` to the actual workspace
when launching child processes; OpenCode uses it to select the task directory.
Use the original agent tools and system prompts. Supply the exact task as
`PROMPT`, and measure each CLI invocation externally, for example with
`/usr/bin/time -f '%e s'`. Save the generated HTML and native session records.

## Pi configuration

Set `PI_CODING_AGENT_DIR` to an external directory containing `models.json`:

```json
{
  "providers": {
    "lia-local": {
      "baseUrl": "http://127.0.0.1:18473/v1",
      "api": "openai-completions",
      "apiKey": "local-only",
      "models": [{
        "id": "qwen3.8-27b",
        "name": "Local Qwen",
        "reasoning": true,
        "thinkingLevelMap": {"high": "xhigh", "xhigh": "xhigh", "max": "xhigh"},
        "contextWindow": 49152,
        "maxTokens": 36864,
        "input": ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "samplingParams": {"temperature": 1, "top_p": 0.95, "top_k": 20, "seed": 42}
      }]
    }
  }
}
```

Run from its empty workspace:

```sh
pi --provider lia-local --model qwen3.8-27b --thinking xhigh \
  --mode json --no-extensions --no-skills --no-context-files \
  --session-dir "$TEST_ROOT/pi-sessions" --print "$PROMPT"
```

## DeepSeek Harness configuration

Set `DSH_HOME` to an external directory and `LIA_LOCAL_API_KEY=local-only`.
Save this overlay as an external `lia.yml`:

```yaml
- id: llm-pi-ai
  config:
    providers:
      lia-local:
        apiKeyEnv: LIA_LOCAL_API_KEY
        api: openai-completions
        baseURL: http://127.0.0.1:18473/v1
        reasoning: max
        models:
          - id: qwen3.8-27b
            name: Local Qwen
            contextWindow: 49152
            maxTokens: 36864
            reasoningEfforts:
              max: xhigh
- id: agent-default-model
  config:
    provider: lia-local
    model: qwen3.8-27b
    reasoningEffort: max
```

```sh
dsh headless --patch "$TEST_ROOT/lia.yml" --json "$PROMPT"
```

## OpenCode configuration

Set `LIA_LOCAL_API_KEY=local-only` and `OPENCODE_CONFIG` to this external JSON
file. Use separate `OPENCODE_CONFIG_DIR`, `XDG_DATA_HOME`, `XDG_CACHE_HOME`, and
`XDG_STATE_HOME` directories; set `OPENCODE_DISABLE_MODELS_FETCH=1`.

```json
{
  "model": "lia-local/qwen3.8-27b",
  "share": "disabled",
  "update": "disable",
  "providers": {
    "lia-local": {
      "name": "lia.rs",
      "env": ["LIA_LOCAL_API_KEY"],
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": {"baseURL": "http://127.0.0.1:18473/v1"},
      "compatibility": {"reasoningField": "reasoning_content"},
      "body": {"reasoning_effort": "xhigh", "temperature": 1, "top_p": 0.95, "top_k": 20, "seed": 42},
      "models": {
        "qwen3.8-27b": {
          "name": "Local Qwen",
          "capabilities": {
            "tools": true,
            "input": ["text"],
            "output": ["text"],
            "reasoning": {"supported": true, "effort": ["xhigh"]}
          },
          "limit": {"context": 49152, "output": 36864}
        }
      }
    }
  }
}
```

```sh
opencode run --standalone --auto --model lia-local/qwen3.8-27b \
  --format json "$PROMPT"
```

`--auto` enables unattended tool approval for the isolated test workspace.

## Verification

Check the final response, actual tool results, generated file, and browser
animation. A zero exit code alone does not establish completion. Preserve raw
requests/responses and failed attempts outside Git.

Declare 49,152 context tokens and a 36,864 output cap in every client. The
configured cap shares context with system prompts, tools, history, and thinking;
native clients may lower individual request caps as context fills. All three
first main requests sent 36,864. Auxiliary requests retain native caps. Record
these differences rather than rewriting requests to hide them.

Before the final test, run a separate short readiness prompt and inspect actual
requests for xhigh and effective sampling. Freeze the shared configuration,
then run one full pelican task per client, serially, with fresh services and
independent empty workspaces. Use a common timeout; these runs allowed 3,600
seconds per CLI. Preserve failures and truncation without lowering thinking or
rerolling until a preferred result appears. Earlier mixed-thinking attempts
are retained externally and are not the results displayed here.
