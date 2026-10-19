# Development Guidelines

## Implementation and Repository

- Implement inference and GPU operators independently in Rust using NVIDIA/cuda-rust. Do not delegate model inference to an existing inference engine.
- Work on development branches rather than directly on `main`.
- Use the MIT License and preserve required third-party copyright notices.
- Keep the repository genuinely Rust-only for implementation code. Do not manipulate language classification to hide other implementation languages.
- Keep model weights, official benchmark fixtures, answers, simulator checkouts, run results, logs, temporary files, and secrets outside Git.
- Follow the scope authorized by the user. A request to inspect or discuss code does not authorize implementation changes.
- Create pull requests only when explicitly requested by the user.
- Do not change agreed requirements, acceptance criteria, or validation scope without explicit user authorization.

## Rust Style and Structure

- Do not define ordinary type aliases. Trait-associated types are allowed.
- Use `anyhow::Result` for application-level error handling rather than custom Result aliases, boxed errors, or string errors. Add context that identifies the failed operation.
- Preserve concrete error types required by trait methods and external callbacks.
- Aim for approximately 256 lines per file. This is a guideline, not a hard limit.
- Organize modules by responsibility and data boundaries. Allow modest overruns and keep indivisible data structures intact rather than splitting mechanically by line count.
- Keep unit tests in an inline `#[cfg(test)] mod tests` within the source file they test. Place integration tests in the repository-root `tests/` directory.
- Keep test-only validation code out of production builds.

## Dependencies and Changes

- Prefer actively maintained dependencies. Evaluate repository activity, release history, compatibility, and actual need rather than relying on release dates alone.
- Keep changes focused and validate them so behavior changes remain attributable.
- Run meaningful checks appropriate to the affected behavior; successful compilation alone does not establish correctness.

## Validation and Reporting

- Define measurement boundaries and token counts consistently when measuring and reporting performance.
- Report only milestones supported by actual validation. Distinguish implementation progress from verified results.
