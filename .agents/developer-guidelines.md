# Coding Principles

Guidelines for production code in ModelOpt. Key values: simplicity, modularity,
and conciseness.

## Principles

- **Prefer simple, surgical changes.** Touch only what the task requires. Avoid speculative
  refactors, broad rewrites, and "while we're here" cleanups.
- **Design for simplicity and readability.** Choose the design that is easiest to understand and maintain.
  Code is read top to bottom: put high-level behavior first, hide lower-level details behind well-named helpers,
  and treat heavy branching as a signal to reconsider the design.
- **Prefer modular, composable solutions.** Avoid input-specific or case-specific hard-coding.
  Use existing extension points when they fit. If none fit, add a simple, focused helper,
  class, or plugin that cleanly captures the new behavior. Keep scope limited to known cases.
- **Respect inheritance boundaries.** Parent abstractions should define shared contracts and
  shared behavior, not child-specific special cases.
- **Don't repeat yourself; keep a single source of truth.** Consolidate repeated logic or intent with a shared helper, API,
  or abstraction when doing so keeps the design simpler. Avoid parallel implementations that can drift out of sync.
- **Comment cautiously.** Comments should add context, not translate code into English.
  Prefer making the code self-explanatory first. Use comments only for non-obvious
  intent or constraints that remain unclear from the code. Apply this guidance to new
  comments only; do not rewrite or delete existing comments just for style.
- **Document public APIs.** Public and higher-level APIs should have docstrings, including examples when useful.
  Internal helpers should usually be self-documenting through clear names and structure.
- **Fix the bug cause, not the side effect.** For bug fixes, find the root cause instead of patching for its side effect.
- **Validate external input once.** Check types and values at the interface boundary. Internal code can trust those
  checks and avoid redundant assertions.
- **Remove dead code.** Delete unused imports, unreachable branches, and obsolete helpers.
- **Use relative paths** from the repo root in commands and file references.

## Testing

- **Develop with focused tests.** During development, write as many focused
  tests as needed, including lower-level unit tests or internal probes, to
  understand and harden behavior.
- **Curate production tests and keep them lean.** Before staging or committing,
  decide which tests should be checked in. Checked-in tests should document
  expected behavior, protect against regressions, or flag backward-incompatible
  behavior changes. Remove redundant lower-level tests when a higher-level test
  already covers the same behavior, keeping CI/CD fast and lean.

## Performant AI Code

- **Keep tensor work on the GPU and avoid unnecessary CPU-GPU syncs.** Reading metadata such as `tensor.shape` is fine.
  Avoid Python scalar extraction and operators such as `tensor.item()`, `float(tensor)`, or `min(tensor)` because they
  can trigger CPU-GPU syncs. Use PyTorch tensor ops such as `tensor.min()` by default, and only extract Python scalars
  when the CPU needs the value. Tensor-value-based Python branching can also break CUDA graphs.
- **Develop with distributed processing in mind.** Examples: Use `print_rank_0` or `warn_rank_0`
  when possible to avoid noisy logs. Guard shared side effects, such as
  file writes or shared state updates, against race conditions between ranks.

## Compatibility

- **Preserve config and checkpoint backward compatibility.** ModelOpt checkpoints include serialized
  `ModeloptBaseConfig` instances such as `QuantizeConfig`. If these Pydantic-based configs change
  without backward compatibility handling, older checkpoints may no longer load. Make breaking changes
  explicit and intentional.
