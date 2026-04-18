# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

NVIDIA Model Optimizer (ModelOpt): open-source library for model optimization techniques including
quantization, pruning, distillation, sparsity, and speculative decoding to accelerate inference.
Primarily Python codebase with optional C++/CUDA extensions supporting PyTorch, ONNX, and Hugging Face/Megatron models.

> If a `CLAUDE.local.md` file exists alongside this file, read and respect it — it contains
> developer-specific overrides that supplement this shared guidance.

## Rules (Read First)

**CRITICAL (YOU MUST):**

- NVIDIA Apache 2.0 license header on ALL new Python/C++/CUDA files — use the SPDX format from `LICENSE_HEADER` (auto-inserted by pre-commit for most files, but must be added manually for files copied from third-party sources, which are excluded from the hook)
- `git commit -s -S` (DCO sign-off + cryptographic signing required). Never attribute AI tools in
  sign-off line
- `pre-commit` hooks run on commit — if files are modified by hooks, re-stage and commit again
- PRs require CODEOWNERS review (auto-assigned based on `.github/CODEOWNERS`)
- After rebasing, always re-run tests locally before pushing
- All code must follow the security guidelines in `SECURITY.md` — violations are blocked as pre-merge errors
- For contribution guidelines, commit conventions, and PR requirements, see `CONTRIBUTING.md`
- New PIP dependencies require license verification — non-permissive licenses need justification and approval from `@NVIDIA/modelopt-setup-codeowners`

## Common Commands

| Task | Command |
|------|---------|
| Install (editable + dev) | `pip install -e ".[dev]"` |
| Enable pre-commit hooks | `pre-commit install` |
| CPU unit tests | `python -m pytest tests/unit` |
| GPU unit tests | `python -m pytest tests/gpu` |
| Megatron GPU tests | `python -m pytest tests/gpu_megatron` |
| TRT-LLM GPU tests | `python -m pytest tests/gpu_trtllm` |
| Single test file | `python -m pytest tests/unit/torch/quantization/test_quant_config.py` |
| Pattern match | `pytest tests/unit -k "test_quantize"` |
| Lint + format (all files) | `pre-commit run --all-files` |
| Lint (diff only) | `pre-commit run --from-ref origin/main --to-ref HEAD` |
| Run via nox (CPU unit) | `nox -s "unit-3.12(torch_211, tf_latest)"` |
| Build docs | `nox -s docs` |
| Build wheel | `nox -s build_wheel` |

## Architecture

ModelOpt code base is organized into four top-level namespaces:

| Namespace | Path | Role |
|-----------|------|------|
| `modelopt.torch` | `modelopt/torch/` | Core PyTorch optimization library |
| `modelopt.onnx` | `modelopt/onnx/` | ONNX model quantization and export |
| `modelopt.deploy` | `modelopt/deploy/` | Deployment utilities for LLMs |
| `modelopt.recipe` | `modelopt/recipe/` | Recipe loading, parsing, and validation infrastructure |

### `modelopt.torch` Sub-packages

| Sub-package | Path | Role |
|-------------|------|------|
| `opt` | `modelopt/torch/opt/` | Core optimization infrastructure (modes, config, state dicts) |
| `quantization` | `modelopt/torch/quantization/` | PTQ, QAT, and quantization-aware algorithms |
| `prune` | `modelopt/torch/prune/` | Structured and unstructured pruning |
| `distill` | `modelopt/torch/distill/` | Knowledge distillation |
| `sparsity` | `modelopt/torch/sparsity/` | Weight and activation sparsity |
| `speculative` | `modelopt/torch/speculative/` | Speculative decoding (Medusa, EAGLE, etc.) |
| `nas` | `modelopt/torch/nas/` | Neural architecture search |
| `export` | `modelopt/torch/export/` | Checkpoint export for TRT-LLM / Megatron |
| `peft` | `modelopt/torch/peft/` | QLoRA and PEFT integration |
| `_deploy` | `modelopt/torch/_deploy/` | Internal deployment utilities |
| `utils` | `modelopt/torch/utils/` | Shared utilities and plugin infrastructure |

### Core Abstraction: Modes

A **mode** is the unit of model optimization in ModelOpt. Each algorithm (quantization, pruning,
etc.) is implemented as one or more modes. Modes are recorded in the model's `modelopt_state` so
optimization workflows can be composed, saved, and restored.

The main entry points are in `modelopt/torch/opt/conversion.py`:
- `apply_mode(model, mode, ...)` — applies an optimization mode to a model
- `restore(model, ...)` — restores a model to a previously saved optimization state
- `save(model, ...)` / `modelopt_state(model)` — captures the current optimization state

### Core Abstraction: Recipes

A **recipe** is a declarative YAML specification of an optimization configuration. Recipes decouple optimization specs from code, enabling reuse, sharing, and version control.

**Built-in recipes** (`modelopt_recipes/`):

- `general/ptq/` — general-purpose PTQ recipes
- `configs/` — shared configuration units referenced by recipes

## Key Files

| File | Role |
|------|------|
| `modelopt/torch/opt/mode.py` | Base class for all optimization modes |
| `modelopt/torch/opt/config.py` | Configuration system for modes |
| `modelopt/torch/opt/conversion.py` | `apply_mode()` / `restore()` entry points |
| `modelopt/torch/quantization/__init__.py` | PTQ/QAT public API |
| `modelopt/torch/export/unified_export_hf.py` | Unified HF checkpoint export |
| `modelopt/torch/export/model_config_export.py` | TRT-LLM model config export |
| `modelopt/deploy/llm/` | LLM deployment utilities |
| `modelopt/recipe/loader.py` | `load_recipe()` / `load_config()` public API |
| `modelopt/recipe/config.py` | Recipe Pydantic models (`ModelOptPTQRecipe`, `RecipeType`) |
| `modelopt_recipes/general/ptq/` | Built-in PTQ recipe YAML files |
| `pyproject.toml` | Optional dependency groups (`[onnx]`, `[hf]`, `[all]`, `[dev]`); ruff, mypy, pytest, bandit, and coverage config |
| `.pre-commit-config.yaml` | Pre-commit hooks (ruff, mypy, clang-format, license headers) |
| `noxfile.py` | Test session definitions |

## Design Patterns

| Pattern | Key Points |
|---------|------------|
| **Mode composition** | Optimization algorithms are composed as sequences of modes, each recorded in `modelopt_state` |
| **Plugin system** | Optional integrations (HuggingFace, Megatron, etc.) loaded lazily via `import_plugin()` |
| **Optional dependencies** | Features gated by install extras (`[onnx]`, `[hf]`, `[all]`); avoid hard imports at module level |
| **Config dataclasses** | Each mode has a typed config; use Pydantic or dataclass conventions |
| **State dict** | Models carry `modelopt_state` for checkpoint save/restore across optimization steps |
| **Declarative recipes** | YAML-based optimization specs in `modelopt_recipes/`; loaded via `load_recipe()`, passed to the model optimization system |

## CI / Testing

| Layer | Location | Notes |
|-------|----------|-------|
| CPU unit tests | `tests/unit/` | Fast, no GPU needed; run in pre-merge CI |
| GPU unit tests | `tests/gpu/` | Requires CUDA GPU |
| Megatron GPU tests | `tests/gpu_megatron/` | Requires Megatron-Core + GPU |
| TRT-LLM GPU tests | `tests/gpu_trtllm/` | Requires TensorRT-LLM + GPU |
| Example/integration tests | `tests/examples/` | Integration tests for examples; see `tests/examples/README.md` |
| Pre-commit / lint | `.pre-commit-config.yaml` | ruff, mypy, clang-format, license headers, bandit |
| Coverage | `pyproject.toml` | 70% minimum on `modelopt/*` |
