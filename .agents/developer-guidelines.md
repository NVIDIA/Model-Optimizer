# Coding Principles

Guidelines for production code in ModelOpt. Key values: simplicity, minimalism,
and elegance.

## Principles

- **Be surgical.** Touch the code required to solve the actual problem, whether
  that is one line or a broader design change. Avoid speculative refactors,
  drive-by cleanup, unrelated rewrites, and half-finished implementations.
- **Fix root causes.** Prefer the right fix over the most local patch. Do not
  paper over symptoms with temporary fixes unless the temporary nature and
  follow-up are explicit.
- **Design for simplicity.** Choose the design that keeps code easiest to read
  and change. Put behavior at the right level, tie extensibility to known needs,
  and treat heavy branching or conditional logic as bad design smells.
- **Respect ownership.** Keep behavior in the layer that owns it. Parent
  abstractions should contain shared contracts and shared behavior, not
  child-specific special cases.
- **Keep one source of truth.** Put shared behavior, configuration, constants,
  validation, and documentation in the single place that owns them. Reuse
  existing helpers and shared APIs instead of copying logic or duplicating
  state.
- **Abstract to simplify.** Use helpers, base classes, registries, adapters,
  plugins, or extension points when they remove real duplication, clarify
  ownership, support current variation, or make call sites simpler. Do not add
  abstractions for speculative future cases.
- **Make code readable at the point of use.** Names, types, and structure should
  make intent clear. Keep high-level orchestration clear, move low-level
  mechanics into well-named helpers when helpful, and put critical code before
  helper details when local conventions allow it.
- **Comment cautiously.** Code should be clear and be the source of truth
  for what happens, how it happens, and why; use comments only when the why is
  not obvious from the code. First ask whether better names, clearer structure,
  or simpler code can explain the intent without a comment. (Apply this guidance
  to new comments only; do not rewrite or delete existing comments.)
- **Scale documentation to the API.** Higher-level and user-visible APIs deserve
  useful docstrings, including examples when helpful. Lower-level internals need
  docstrings only when names, types, and structure are not enough.
- **Validate at boundaries.** Check user input, files, network responses, and
  external API results at the edge. Keep internal code simple by trusting types
  and invariants instead of repeatedly checking for impossible states.
- **Remove touched dead code.** Delete unused imports, unreachable branches,
  obsolete placeholders, stale TODOs, and debug code when they are part of the
  behavior you are already touching.
- **Use workspace-relative paths.** Use relative paths in commands and file
  references unless an absolute path is needed to disambiguate.

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

- **Avoid stray CPU-GPU syncs.** Tensor metadata such as `tensor.shape` is safe
  to read, but scalar extraction or CPU transfers such as `tensor.item()`,
  `float(tensor)`, `bool(tensor)`, `tensor.cpu()`, `tensor.numpy()`, etc. can
  force CPU-GPU synchronization. Keep computation on GPU unless the CPU actually
  needs the value.
- **Use rank-aware logging.** Default to `print_rank_0` instead of `print` and
  `warn_rank_0` instead of generic warnings. Use per-rank output only when each
  process needs to report distinct state. Generic prints and warnings clog
  distributed logs.
- **Respect distributed invariants.** Avoid hidden synchronization, global state,
  per-rank file races, or assumptions that only hold on a single process.

## Compatibility

- **Preserve config and checkpoint compatibility.** Treat ModelOpt config schemas
  and checkpoint formats as persisted contracts. When changing configs such as
  `QuantizeConfig`, maintain backward compatibility with previous ModelOpt
  checkpoints unless a breaking change is explicit and intentionally handled.
