# ModelOpt code review guidelines

ModelOpt-specific criteria for reviewing pull requests in this repository. They apply on top of the
general review prompt and the conventions in `AGENTS.md` and `CONTRIBUTING.md`.

ModelOpt is NVIDIA's open-source library for model optimization (quantization, pruning, distillation,
sparsity, speculative decoding, NAS, PEFT) targeting PyTorch / ONNX / HuggingFace / Megatron, with
deployment into TRT-LLM / vLLM / SGLang. Apply general correctness reasoning to whichever
`modelopt/torch/<sub-package>` the diff touches: you already know the algorithms (quantization
formulas, distillation losses, N:M sparsity mask selection, speculative draft-token alignment, ...).
These guidelines cover the ModelOpt-specific structural concerns the diff alone does not show.

## Division of labor with CodeRabbit

CodeRabbit (`.coderabbit.yaml`) already reviews every PR and gates the security anti-patterns. Do
not duplicate it. Skip:

- Style, formatting and naming nits (ruff and CodeRabbit cover them), and simple typos.
- The security anti-patterns listed in `.coderabbit.yaml`: `torch.load(weights_only=False)` or
  `numpy.load(allow_pickle=True)` without justification, hardcoded `trust_remote_code=True`,
  `eval`/`exec` on external input, `# nosec` bypasses, non-permissive new PIP dependencies.
- Generic "consider adding a test" suggestions for trivial changes.

Your value is what pattern matching cannot do: tracing dataflow across files, reasoning about
mode/state composition, judging export compatibility, and catching algorithm-level bugs.

## Investigation budget

The review is time-boxed. Every tool round-trip is a separate rate-limited request, and throttling has
caused timeouts before. Aim for one pass that surfaces everything the author needs, in about 25 tool
calls: a review that runs out of time posts nothing.

- Batch independent reads into one turn, for example the patches, `AGENTS.md`, `CONTRIBUTING.md` and
  `LICENSE_HEADER` together.
- Review in this order: `modelopt/`, then `examples/`, then `tests/`. Skip config, lock, generated,
  docs and data files unless the change there is the point of the PR.
- Read the patches of `modelopt/` and `examples/` first, and those of `tests/` only if budget remains.
- A patch already shows the changed lines, so do not re-read them. Read a file only for context the
  patch lacks, or when mode/state composition requires it, and then prefer the changed hunk plus about
  40 lines over the whole file. Do not re-read files already in context.
- Over 50 changed files, open at most about 15 source files, highest risk first. Say in the summary how
  many files changed, which you reviewed, and which paths you did not open.
- Only if the diff touches mode registration, config schema or a public `__init__.py` export of a
  `modelopt/torch/<sub-package>`, read that sub-package's `__init__.py`, `mode.py` and `config.py`.
- Trace only new or renamed public symbols (functions, arguments, config fields, exported names): grep
  call sites in `modelopt/`, and in `tests/` and `examples/` only if that is inconclusive. Bugs
  concentrate where a symbol meets its caller (mode registration, export paths, restore logic).
- Trace the algorithm end to end through the diff. Confirm each new variable, argument or field has a
  real runtime use, not just a declaration, a docstring or a discard assignment (`_ = new_arg`).

## Critical issues (must fix)

### Algorithm correctness

- The implementation matches the intended technique: quantization scales, rounding and saturation,
  distillation loss composition, sparsity mask selection, pruning importance scoring, speculative draft
  acceptance, NAS supernet weight sharing, and so on.
- Silent numerical bugs: a missing fp32 upcast in reductions, a wrong reduction dimension, a missing
  division-by-zero guard, casts that wrap instead of saturate, gradient flow through stop-gradient
  boundaries.
- State corrupted across calibration, search or training passes. Statistics left over from a previous
  run are a common foot-gun.

### Mode and state composability

- **Mode registration:** a new mode registers correctly with `apply_mode()` / `restore()`, declares its
  dependencies, and produces a `modelopt_state` entry that round-trips through save and restore.
- **State dict schema:** a changed `modelopt_state` schema needs a migration path or a version bump.
  Silently changing keys breaks restore for existing checkpoints.
- **Restore fidelity:** after `restore(model, state)` the model is functionally identical to the saved
  one. Module replacements, hooks and parameters are re-applied.
- **Plugin laziness:** optional integrations (HF, Megatron, TRT-LLM, ONNX) are not hard-imported at
  module load. Gate them behind `import_plugin()` so users without those extras do not break.

### Export compatibility

- HF export (`unified_export_hf.py`) produces a checkpoint that loads cleanly in transformers and
  matches the on-device dtype.
- TRT-LLM export (`model_config_export.py`) emits a valid `config.json` with the correct `quant_algo`,
  `kv_cache_quant_algo`, scale tensor names and weight layout.
- ONNX export uses opsets and operator versions supported by the target consumer (TRT, ORT).

## Important issues (should fix)

### Backward compatibility

- Renamed or removed arguments or config fields without a deprecation path break existing user scripts.
- `modelopt_recipes/*.yaml` schema changes without a version bump make old recipes misparse silently.
- Changed defaults silently alter behavior for users relying on them.
- Changed signatures, return types or side effects in `modelopt/torch/*/__init__.py` (public API)
  without a backward-compatible shim.
- Modified `modelopt_state` keys or structure without migration make existing optimized checkpoints
  unloadable.

### Performance

- Unnecessary CPU-GPU synchronization in hot paths: `.item()`, `.cpu()`, `torch.cuda.synchronize()`,
  Python-side tensor value checks.
- Memory regressions: double-allocating weights, holding tensors past their lifetime.

## Suggestions (nice to have)

Suggestions are non-blocking. Raise them when genuinely useful and skip nits that are not.

- Stale, imprecise or misleading comments and docstrings. A wrong docstring is worse than none.
- Missing shape or dtype assertions at module or parallelism boundaries where they would catch real
  bugs.
- Functions mixing many unrelated responsibilities that would benefit from splitting.

## Comment format and verdict

Prefix each inline comment with a severity and category tag:

- `**[CRITICAL Algorithm]**`, `**[CRITICAL ModeState]**`, `**[CRITICAL Export]**`
- `**[IMPORTANT Compatibility]**`, `**[IMPORTANT Performance]**`
- `**[SUGGESTION]**`

For each finding explain what the issue is, why it matters (impact or risk) and a specific fix. Use an
inline `suggestion` block only for simple, self-contained line replacements (typos, renames,
single-line fixes). For structural changes that add, remove or reorganize blocks of code, describe the
change in the summary or the comment body with a code block instead.

Any CRITICAL or IMPORTANT finding makes the decision `comment`. SUGGESTIONs alone never block `approve`.
