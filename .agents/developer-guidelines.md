# Minimal-Developer Principles

Guidelines for writing production code. Key values: simplicity, minimalism, and
elegance.

## Philosophy

- **Be Surgical, Minimal edits.** Touch only what the task requires. No speculative
  refactors, drive-by cleanup, or half-finished implementations.
- **Good design simplifies code.** Good architecture makes code easier to read
  and easier to change. Use abstractions that put code at the right level, and
  keep designs extensible for known needs without over-engineering for
  speculative future cases. Heavy branching and conditional logic are design
  smells because they are hard to read and prone to bugs.
- **Respect ownership and layering.** Keep behavior in the layer that owns it.
  Parent abstractions should contain shared contracts and shared behavior, not
  child-specific special cases.
- **Prefer clean extension over branching.** When behavior varies, use explicit
  extension points such as strategies, adapters, callbacks, or overrides. Update
  a parent or shared base when it is the right extension point, but do not add
  extension points for speculative future cases.
- **Use helpers for duplication and right-level abstraction.** Add helpers when
  they remove duplicated code across multiple places or let higher-level code
  use helpful names while low-level details stay abstracted away.
- **Use meaningful names.** Methods, variables, objects, and helpers should make
  intent clear at the point of use.
- **Code should read like prose.** Well-written code is self-explanatory and
  elegant. Keep high-level orchestration clear, and move low-level mechanics
  into well-named helpers when that improves readability.
- **Critical code first.** Order files so the most important or user-facing code
  appears before helper details when local conventions allow it.
- **Validate at boundaries, then trust invariants.** Check untrusted inputs at system boundaries such as user input, external APIs, files, networks, and process boundaries. After validation, internal code should trust its types and invariants instead of adding defensive checks for states that should be impossible.
- **Comments explain why.** Code is the source of truth for what happens and
  how. Add comments only when the reason is not obvious. Redundant comments are
  a maintainability burden. If a comment feels necessary, first check whether
  better design or naming would make the code explain itself.
- **Preserve existing comments.** Minimalism applies to new comments; do not
  delete existing comments casually. Production code is shared by many
  developers, and unnecessary changes to others' code create avoidable review and
  approval overhead.
- **Docstrings scale with API level.** Higher-level and user-visible APIs
  deserve useful docstrings, including examples when helpful. Lower-level
  internals should use minimal docstrings, or none, when well-named identifiers
  are enough.
- **Remove dead code.** Delete unused imports, unreachable branches, and
  obsolete placeholders.
- **Use workspace-relative paths** in commands and file references unless an
  absolute path is needed to disambiguate.

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
  `float(tensor)`, `bool(tensor)`, `tensor.cpu()`, and `tensor.numpy()` can force
  CPU-GPU synchronization. Keep computation on GPU unless the CPU actually needs
  the value.
- **Use rank-aware logging.** Default to `print_rank_0` instead of `print` and
  `warn_rank_0` instead of generic warnings. Use per-rank output only when each
  process needs to report distinct state. Generic prints and warnings clog
  distributed logs.
