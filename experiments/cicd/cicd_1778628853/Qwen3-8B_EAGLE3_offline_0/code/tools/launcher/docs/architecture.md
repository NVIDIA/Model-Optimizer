# Architecture

## Shared Core

The launcher is built on `core.py`:

```text
core.py
├── Dataclasses: SandboxTask, SandboxPipeline, GlobalVariables
├── Executor builders: build_slurm_executor(), build_docker_executor()
├── Job runner: run_jobs()
├── Version reporter: report_versions()
├── Factory registry: register_factory(), set_slurm_config_type()
└── Default env: get_default_env()

launch.py
├── imports core.py
├── slurm_config.py (env-var driven)
├── registers: slurm_factory
├── packager (LAUNCHER_DIR relative)
└── launch() entrypoint
```

## Code Packaging

`PatternPackager` creates a tar.gz of source code and rsyncs it to the cluster. The `code/` directory mirrors the launcher structure:

```text
code/
├── modules/
│   ├── Megatron-LM/megatron/...
│   └── Model-Optimizer/modelopt/...
└── common/
    ├── megatron_lm/quantize/quantize.sh
    ├── tensorrt_llm/query.sh
    ├── vllm/query.sh
    ├── eagle3/
    └── query.py
```

## ModelOpt Mount Mechanism

The container image ships with pre-installed ModelOpt. The launcher **bind-mounts your local `modelopt/` over this path**, so local changes take effect without rebuilding the container.

Configured via `modelopt_install_path` in `SlurmConfig`:

```yaml
slurm_config:
  modelopt_install_path: /usr/local/lib/python3.12/dist-packages/modelopt
```

At runtime:

- **Slurm**: `{job_dir}/{experiment_title}/{exp_id}/{task}/code/modules/Model-Optimizer/modelopt` → `{modelopt_install_path}`
- **Docker**: `{LAUNCHER_DIR}/modules/Model-Optimizer/modelopt` → `{modelopt_install_path}`

Find the install path for a given container:

```bash
docker run --rm <image> python3 -c "import modelopt; print(modelopt.__file__)"
```

## Model-Optimizer Symlink

`tools/launcher/modules/Model-Optimizer` is a **symlink** to `../../..` (the Model-Optimizer root), not a submodule. This avoids recursive nesting.

- Git tracks the symlink natively (`git clone` preserves it)
- `launch.py` auto-creates the symlink on first run if missing
- The packager's `find` follows symlinks

## Factory System

YAMLs reference a factory by name:

```yaml
slurm_config:
  _factory_: "slurm_factory"
  nodes: 1
```

Factories are registered at import time via `register_factory()`. In `launch.py`, `slurm_factory` reads from environment variables. In `slurm.py`, it resolves to a cluster-specific factory based on `SLURM_CLUSTER`:

```bash
SLURM_CLUSTER=cw_dfw uv run slurm.py --yaml config.yaml --yes
```

## Global Variables

Pipeline YAMLs support `<<global_vars.X>>` interpolation:

```yaml
pipeline:
  global_vars:
    hf_local: /hf-local/

  task_0:
    environment:
      - HF_MODEL_CKPT: <<global_vars.hf_local>>Qwen/Qwen3-8B
```

Resolved in `SandboxPipeline.__post_init__` using regex substitution.

## Typed Task Classes

`SandboxTask` is generic (script/args/environment). Typed tasks add structured configs:

```python
@dataclass
class MegatronLMQuantizeConfig:
    model: str = "Qwen/Qwen3-8B"
    quant_cfg: str = "NVFP4_DEFAULT_CFG"
    tp: int = 4
    hf_local: str = "/hf-local/"

@dataclass
class MegatronLMQuantizeTask(SandboxTask):
    config: MegatronLMQuantizeConfig = None

    def __post_init__(self):
        if self.config:
            self.script = "common/megatron_lm/quantize/quantize.sh"
            self.args = [f"--calib-size {self.config.calib_size}", ...]
            self.environment = [{"TP": str(self.config.tp)}, ...]
```

YAML usage:

```yaml
task_0:
  _target_: common.megatron_lm.quantize.task.MegatronLMQuantizeTask
  config:
    model: Qwen/Qwen3-8B
    tp: 4
  slurm_config:
    _factory_: "slurm_factory"
```

### Adding a new typed task

1. Create `common/<workflow>/task.py` alongside the shell script
2. Define a config dataclass with tunable parameters and defaults
3. Inherit from `SandboxTask`, convert config → script/args/environment in `__post_init__`
4. Reference via `_target_` in YAML

Future structure:

```text
common/
├── megatron_lm/quantize/task.py    # MegatronLMQuantizeTask
├── megatron_lm/train/task.py       # MegatronLMTrainTask (future)
├── eagle3/task.py                  # Eagle3OfflineTask (future)
└── tensorrt_llm/task.py            # TRTLLMQueryTask (future)
```

## Metadata

Each experiment writes `metadata.json` to `experiments/<title>/<id>/`:

```json
{
  "experiment_id": "cicd_1773420387",
  "job_name": "Qwen3-8B_NVFP4_DEFAULT_CFG",
  "allow_to_fail": false,
  "note": ""
}
```
