# Agent Instructions for ModelOpt

These instructions apply to AI-assisted work in this repository. Keep
authoritative details in the existing ModelOpt docs instead of duplicating them
here.

- **Development values:** Read and follow `.agents/developer-guidelines.md` for
  code development in ModelOpt.

- **Project overview and install:** Start with `README.md`.

- **Code areas:** `modelopt.torch.opt` is shared optimization infrastructure;
  `modelopt.torch.quantization` covers quantization algorithms;
  `modelopt.torch.export` covers export flows; `modelopt.torch.prune`,
  `modelopt.torch.distill`, `modelopt.torch.sparsity`, and
  `modelopt.torch.speculative` cover other torch algorithms. `modelopt.onnx`,
  `modelopt.deploy`, and `modelopt.recipe` cover ONNX, deployment, and recipes.

- **Contributing and PRs:** Read `CONTRIBUTING.md` for commit conventions,
  DCO sign-off, signing, PR expectations, and review requirements.

- **Security:** Read `SECURITY.md` before changing external input handling,
  serialization, subprocess use, dependencies, or user-facing behavior.

- **Dependencies:** Read `pyproject.toml` for package metadata and optional
  extras. New pip dependencies require owner license review.

- **License headers:** New Python, C++, and CUDA files need the SPDX license
  header from `LICENSE_HEADER`.

- **Running tests:** Use `pyproject.toml`, `noxfile.py`, and `tests/` to choose
  focused tests for the changed area.

- **Running pre-commit:** Use `.pre-commit-config.yaml` for lint, format, type,
  license, and security hooks. Hooks may modify files; review and re-stage those
  changes before committing.
