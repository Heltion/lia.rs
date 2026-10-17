# Benchmark Specification

## Procedure

Compare lia.rs and llama.cpp on the same hardware and model file, with thinking
and built-in MTP enabled and verified in both. Use a pinned official BFCL v3
revision for tasks, tools, simulation, and scoring.

Select three different tasks before testing. For each task, run llama.cpp then
lia.rs serially, completing all turns and tool calls. Start a fresh service and
independent conversation and simulator state for each run; allow normal cache
reuse within a task. Report official correctness separately from performance.

## Measurements

Record three times in seconds and two token counts for each complete task:

| Measurement | Definition |
| --- | --- |
| Preparation time | From service launch until ready for inference, including model loading |
| Prefill time | Sum of prefill durations across all requests |
| Decode time | Sum of decode durations across all requests, including MTP drafting, verification, and recovery |
| Prefill tokens | Input tokens actually processed, excluding reused cached tokens |
| Decode tokens | Final output tokens produced during decode, including thinking, excluding rejected drafts and tokens produced by prefill |

Use equivalent phase boundaries and counting rules in both implementations.
Measure preparation externally with a monotonic clock and obtain phase times
from the servers. Missing measurements are not zero.

## Calculation and Reporting

```text
prefill tok/s  = prefill tokens / prefill time
decode tok/s   = decode tokens / decode time
combined tok/s = (prefill tokens + decode tokens)
                 / (preparation time + prefill time + decode time)

preparation ratio = llama.cpp preparation time / lia.rs preparation time
prefill ratio     = lia.rs prefill tok/s / llama.cpp prefill tok/s
decode ratio      = lia.rs decode tok/s / llama.cpp decode tok/s
combined ratio    = lia.rs combined tok/s / llama.cpp combined tok/s
```

Report all measurements, three speeds for each implementation, and four ratios.
Ratios above 1 favor lia.rs. Use combined speed as the primary optimization
metric; this specification sets no target. Use N/A for undefined rates.

Combined speed excludes tool execution, scoring, and overhead outside the
measured phases; it is not end-to-end task wall time. Retain token counts so
differences in generated trajectories and cache reuse remain visible.

Report each task separately. For a suite summary, sum times and counts for
each implementation before calculating speeds and ratios; do not average
per-task speeds or ratios.

## Records

Record code and binary identities, model hash, benchmark revision, selected
tasks, hardware, software, launch configuration, and relevant differences.
Keep conditions comparable and disclose limitations. Preserve failures,
retries, and original measurements. Keep run records, official fixtures,
models, temporary files, and secrets outside Git.
