# Coding Principles

Guidelines for production code in ModelOpt. Key values: simplicity, minimalism,
and elegance.

## Principles

- **Design for simplicity.** Before planning non-trivial changes, choose the
  design that keeps code easiest to read and change. Put behavior at the right
  level, tie extensibility to known needs, and treat heavy branching or
  conditional logic as bad design smells.
- **Be surgical.** Touch the code required to solve the actual problem, whether
  that is one line or a broader design change. Avoid speculative refactors,
  drive-by cleanup, unrelated rewrites, and half-finished implementations.
- **Fix root causes.** Prefer the right fix over the most local patch. Do not
  paper over symptoms with temporary fixes unless the temporary nature and
  follow-up are explicit.
- **Respect ownership and layering.** Keep behavior in the layer that owns it.
  Parent abstractions should contain shared contracts and shared behavior, not
  child-specific special cases.
- **Use abstractions to simplify.** Add helpers, base classes, registries,
  adapters, plugins, or other abstractions when they remove real duplication,
  clarify ownership, or put behavior at the right level. Do not add abstractions
  for speculative future cases.
- **Prefer extension over branching.** When behavior varies, use explicit
  extension points such as adapters, registries, callbacks, plugins, or
  overrides. Update a parent or shared base when it is the right extension
  point. Do not add extension points for speculative future cases.
- **Make code readable at the point of use.** Names, types, and structure should
  make intent clear. Keep high-level orchestration clear, and move low-level
  mechanics into well-named helpers when that improves readability.
- **Put critical code first.** Order files so the most important or user-facing
  code appears before helper details when local conventions allow it.
- **Validate outside input once.** Check user input, files, network responses,
  and external API results at the edge. Keep internal code simple instead of
  repeatedly checking for impossible states.
- **Comments explain why, concisely.** Code is the source of truth for what happens and
  how. Add comments only when the reason is not obvious. Redundant comments are
  a maintainability burden. If a comment feels necessary, first check whether
  better design or naming would make the code explain itself.
- **Apply comment guidance to new comments only.** Use these standards only when adding
  new comments. Do not rewrite or delete existing comments as cleanup;
- **Scale documentation to the API.** Higher-level and user-visible APIs deserve
  useful docstrings, including examples when helpful. Lower-level internals need
  docstrings only when names, types, and structure are not enough.
- **Remove dead code.** Delete unused imports, unreachable branches, obsolete
  placeholders, stale TODOs, and debug code when they are part of the touched
  behavior.
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
