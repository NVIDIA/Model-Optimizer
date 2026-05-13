# Agent Instructions for ModelOpt

These instructions apply to AI-assisted work in this repository. Keep
authoritative details in the existing ModelOpt docs instead of duplicating them
here.

- **Development values:** Read and follow `.agents/developer-guidelines.md` for
  code development in ModelOpt.

- **Project overview and install:** Start with `README.md`.

- **Code areas:** `modelopt.torch.opt` is shared optimization infrastructure;
  `modelopt.torch.quantization` covers quantization algorithms;
  `modelopt.torch.speculative` covers speculative decoding; and
  `modelopt.torch.export` covers export flows. `modelopt.torch.prune`,
  `modelopt.torch.distill`, and `modelopt.torch.sparsity` cover other torch
  algorithms. `modelopt.onnx`, `modelopt.deploy`, and `modelopt.recipe` cover
  ONNX, deployment, and recipes.

- **Contributing and PRs:** Read `CONTRIBUTING.md` for commit conventions,
  DCO sign-off, signing, PR expectations, and review requirements.

- **PR readiness:** Before opening or marking a PR ready for review, read
  `.github/PULL_REQUEST_TEMPLATE.md` and satisfy the checklist.

- **Security:** Read `SECURITY.md` before changing external input handling,
  serialization, subprocess use, dependencies, or user-facing behavior.

- **Dependencies:** Read `pyproject.toml` for package metadata and optional
  extras. New pip dependencies require owner license review.

- **License headers:** New Python, C++, and CUDA files need the SPDX license
  header from `LICENSE_HEADER`.

- **Running tests:** Follow the
  [test instructions](../CONTRIBUTING.md#-writing-and-running-tests) in
  `CONTRIBUTING.md`. For fast initial iteration, choose focused tests for the
  changed area from `tests/`.

- **Running pre-commit:** Follow the
  [pre-commit hook instructions](../CONTRIBUTING.md#pre-commit-hooks) in
  `CONTRIBUTING.md`. Hooks may modify files; review and re-stage those changes
  before committing.

## Git Workflow

- **Never `git push` without explicit approval in the current turn.** Commit
  locally is fine; publishing to a remote is not.
- After `git commit`, stop and wait for the user to
  say "push", "publish", "ship", or equivalent before running `git push`,
  `gh pr create`, or any push-option flags like `-o merge_request.create`.
