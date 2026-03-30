# Contributing

## Adding a New Model

1. Create `examples/<Organization>/<ModelName>/` directory
2. Add a YAML config using a typed task class:

```yaml
job_name: MyModel_NVFP4
pipeline:
  task_0:
    _target_: common.megatron_lm.quantize.task.MegatronLMQuantizeTask
    config:
      model: org/my-model
      quant_cfg: NVFP4_DEFAULT_CFG
      tp: 4
    slurm_config:
      _factory_: "slurm_factory"
      nodes: 1
      ntasks_per_node: 4
      gpus_per_node: 4
```

1. Test with dry run: `uv run launch.py --yaml <path> --dryrun --yes -v`
1. Create a `_local.yaml` variant with `tp: 1` for single-GPU testing

## Adding a New Typed Task

1. Create `common/<workflow>/task.py` alongside the shell script it wraps
2. Define a config dataclass:

```python
@dataclass
class MyWorkflowConfig:
    """Typed config with named fields and defaults."""
    model: str = "default/model"
    param: int = 4
    hf_local: str = "/hf-local/"
```

1. Inherit from `SandboxTask`:

```python
@dataclass
class MyWorkflowTask(SandboxTask):
    config: MyWorkflowConfig = None

    def __post_init__(self):
        if self.config:
            self.script = "common/<workflow>/run.sh"
            self.args = [f"--param {self.config.param}"]
            self.environment = [{"MODEL": self.config.model}]
```

1. Reference via `_target_` in YAML:

```yaml
task_0:
  _target_: common.<workflow>.task.MyWorkflowTask
  config:
    model: org/my-model
```

## Reporting Bugs

Include these three items:

1. **Version summary** — printed at the start of every run:

   ```text
   ============================================================
   Version Report
   ============================================================
     Launcher                       d28acd33     (main)
     Megatron-LM                    1e064f361    (main)
     Model-Optimizer                69c0d479     (main)
   ============================================================
   ```

2. **Reproducible config** — dump with `--to-yaml`:

   ```bash
   uv run launch.py --yaml <config> --to-yaml bug_report.yaml
   ```

3. **Error output** — the relevant traceback from the job log.

File issues at: <https://github.com/NVIDIA/Model-Optimizer/issues>
