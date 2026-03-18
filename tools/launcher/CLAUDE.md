# CLAUDE.md — ModelOpt Launcher

## Overview

The launcher submits ModelOpt quantization, training, and evaluation jobs to Slurm clusters or runs them locally with Docker.

## Key Files

| File | Role |
|------|------|
| `launch.py` | Public entrypoint — accepts `--yaml` or `pipeline=@` |
| `core.py` | Shared dataclasses, executor builders, run loop, version reporting |
| `slurm_config.py` | `SlurmConfig` dataclass and env-var-driven `slurm_factory` |
| `common/` | Shell scripts and `query.py` packaged to the cluster |
| `modules/Megatron-LM/` | Git submodule |
| `modules/Model-Optimizer` | Symlink to `../..` (auto-created by `launch.py` if missing) |

## Common Commands

```shell
# Run locally with Docker
uv run launch.py --yaml examples/Qwen/Qwen3-8B/megatron_lm_ptq.yaml hf_local=/mnt/hf-local --yes

# Run on Slurm (set env vars first)
uv run launch.py --yaml examples/Qwen/Qwen3-8B/megatron_lm_ptq.yaml --yes

# Dry run — preview resolved config
uv run launch.py --yaml examples/Qwen/Qwen3-8B/megatron_lm_ptq.yaml --dryrun --yes -v

# Dump resolved config
uv run launch.py --yaml examples/Qwen/Qwen3-8B/megatron_lm_ptq.yaml --to-yaml resolved.yaml

# Run unit tests
uv pip install pytest
uv run python3 -m pytest tests/ -v
```

## YAML Config Format

The `--yaml` format maps top-level keys to `launch()` function arguments:

```yaml
job_name: Qwen3-8B_NVFP4_DEFAULT_CFG
pipeline:
  global_vars:
    hf_local: /hf-local/
  task_0:
    script: common/megatron_lm/quantize/quantize.sh
    args:
      - --calib-dataset-path-or-name <<global_vars.hf_local>>abisee/cnn_dailymail
    environment:
      - MLM_MODEL_CFG: Qwen/Qwen3-8B
      - HF_MODEL_CKPT: <<global_vars.hf_local>>Qwen/Qwen3-8B
      - TP: 4
    slurm_config:
      _factory_: "slurm_factory"
      nodes: 1
      ntasks_per_node: 4
      gpus_per_node: 4
```

Key conventions:

- Scripts go in `common/` (not `services/`)
- `<<global_vars.X>>` interpolation for shared values across tasks
- `_factory_: "slurm_factory"` — resolved via `register_factory()` in `core.py`
- Environment is list-of-single-key-dicts: `- KEY: value`
- CLI overrides: `pipeline.task_0.slurm_config.nodes=2`

## Architecture

```text
launch.py → imports core.py + slurm_config.py
               ↓
           core.run_jobs()
               ↓
         build_docker_executor() or build_slurm_executor()
               ↓
         nemo_run.Experiment → Docker or Slurm
```

- `set_slurm_config_type(SlurmConfig)` — patches `SandboxTask` annotation at import time
- `register_factory("slurm_factory", slurm_factory)` — enables YAML `_factory_` resolution
- `report_versions(base_dir)` — prints git commit/branch for launcher + submodules
- `get_default_env(title)` — returns `(slurm_env, local_env)` dicts

## Adding a New Model Config

1. Create `examples/<Org>/<Model>/megatron_lm_ptq.yaml` following the format above
2. Set `MLM_MODEL_CFG` to the HuggingFace repo ID
3. Set `QUANT_CFG` (e.g., `NVFP4_DEFAULT_CFG`, `INT8_DEFAULT_CFG`)
4. Set GPU/node counts based on model size
5. Test: `uv run launch.py --yaml <path> --dryrun --yes -v`

## Testing

65 unit tests in `tests/`. Run standalone without installing `modelopt`:

From the launcher directory:

```shell
uv run python3 -m pytest tests/ -v
```

Tests cover: core dataclasses, factory registry, global_vars interpolation, YAML formats, Docker/Slurm executor construction (mocked), environment merging, metadata writing, and end-to-end Docker launch via subprocess.

## Further Reading

- [docs/configuration.md](docs/configuration.md) — YAML formats, overrides, hf_local
- [docs/architecture.md](docs/architecture.md) — Shared core, factory system, typed tasks, mount mechanism
- [docs/testing.md](docs/testing.md) — Running tests locally and in CI
- [docs/claude_code.md](docs/claude_code.md) — Claude Code workflows
- [docs/contributing.md](docs/contributing.md) — Adding models, typed tasks, bug reporting
