# ModelOpt Launcher

Submit ModelOpt quantization, training, and evaluation jobs to Slurm clusters or run them locally with Docker.

## Quick Start

```bash
# Install
curl -LsSf https://astral.sh/uv/install.sh | sh
git submodule update --init --recursive

# Run locally with 1 GPU
cd Model-Optimizer/tools/launcher
uv run launch.py --yaml examples/Qwen/Qwen3-8B/megatron_lm_ptq_local.yaml hf_local=/mnt/hf-local --yes

# Run on a Slurm cluster (4 GPUs)
export SLURM_HOST=login-node.example.com
export SLURM_ACCOUNT=my_account
export SLURM_HF_LOCAL=/mnt/hf-local
export SLURM_JOB_DIR=/shared/experiments
uv run launch.py --yaml examples/Qwen/Qwen3-8B/megatron_lm_ptq.yaml --yes
```

> **Local vs cluster:** `megatron_lm_ptq.yaml` defaults to TP=4 on 4 GPUs.
> Use `megatron_lm_ptq_local.yaml` for single-GPU local Docker runs.

## Directory Structure

```text
tools/launcher/
├── launch.py                       # Main entrypoint
├── core.py                         # Core logic (dataclasses, executors, run loop)
├── slurm_config.py                 # SlurmConfig dataclass and factory
├── common/                         # Scripts and typed tasks
│   ├── megatron_lm/quantize/
│   │   ├── quantize.sh             # PTQ quantization + MMLU evaluation
│   │   └── task.py                 # MegatronLMQuantizeTask (typed config)
│   ├── tensorrt_llm/query.sh       # TRT-LLM server + query
│   ├── vllm/query.sh               # vLLM server + query
│   ├── eagle3/                     # EAGLE3 speculative decoding scripts
│   └── specdec_bench/              # Speculative decoding benchmark
├── examples/                        # Example configs
│   └── Qwen/Qwen3-8B/
│   ├── megatron_lm_ptq.yaml        # PTQ (4 GPUs, Slurm)
│   ├── megatron_lm_ptq_local.yaml  # PTQ (1 GPU, local Docker)
│   └── hf_offline_eagle3.yaml      # EAGLE3 offline pipeline
├── tests/                          # 64 unit tests
├── modules/                        # Dependencies
│   ├── Megatron-LM/                # Git submodule
│   └── Model-Optimizer -> ../..    # Symlink (auto-created)
└── docs/                           # Documentation
    ├── configuration.md            # YAML formats, overrides, hf_local
    ├── architecture.md             # Design, factory system, typed tasks
    ├── testing.md                  # Running tests, CI
    ├── claude_code.md              # Claude Code workflows
    └── contributing.md             # Adding models, bug reporting
```

## Documentation

| Guide | Description |
|-------|-------------|
| [Configuration](docs/configuration.md) | YAML formats, CLI overrides, flags, `hf_local` |
| [Architecture](docs/architecture.md) | Shared core, factory system, typed tasks, mount mechanism |
| [Testing](docs/testing.md) | Running tests locally and in CI |
| [Claude Code](docs/claude_code.md) | Submit, monitor, diagnose workflows |
| [Contributing](docs/contributing.md) | Adding models, typed tasks, bug reporting |

### Bounded concurrent calibration

`SandboxPipeline.max_concurrent_tasks` defaults to 1. Set it explicitly to cap
concurrent Slurm allocations. `SandboxTask.wait_for_all=True` joins the outstanding
tasks and gates subsequent tasks. Each task retains its own `SlurmConfig`, including
nodes, GPUs, host memory and wall-time limits. Local Docker execution stays serial.

For AutoQuant, use five independent tasks with the same full recipe, source model,
seed and calibration dataset, each with a different
`--auto_quantize_calibration_candidate` index and a shared
`--auto_quantize_calibration_dir`. Workers return after calibration. Append one
coordinator with `wait_for_all=True`, the same calibration directory, and
`--auto_quantize_checkpoint`; omit the candidate index. The coordinator validates
all results before scoring and publishing the checkpoint. BF16 needs no task.
Workers exit before KV calibration, casting and export; none remain running after
the coordinator starts. Later tasks run only when explicitly listed in the pipeline.

Use `hf_ptq.py --model <source> --recipe <recipe>
--auto_quantize_list_calibration_candidates` to print the canonical index mapping
without loading weights or requiring a GPU. Set
`--auto_quantize_calibration_max_input_bytes` to bound each worker's CPU input
snapshot (64 MiB by default). Disk-offloaded meta weights are currently unsupported.

```python
pipeline = SandboxPipeline(
    max_concurrent_tasks=2,
    allow_to_fail=True,  # let independent candidates finish even if another fails
    tasks=[*candidate_tasks, coordinator_task],
)
coordinator_task.wait_for_all = True
```

Use the `tasks` list or YAML `task_configs` for this six-task workflow. Bound the
resources of every worker: two concurrent four-GPU tasks reserve eight GPUs, plus
their host memory and separate candidate output space. Do not point workers at the
coordinator checkpoint. On failure, resubmit the same pipeline or only failed/missing
workers; completed candidates are validated and reused. Keep source, inputs,
configuration, process topology and numerical settings identical across retries.
Completed candidates are immutable and reused; replacing the coordinator checkpoint
reclaims its preceding generation under the same lock. Do not read that checkpoint
outside the coordinator while it is being replaced.
Measure both elapsed time and all allocated GPU-hours before increasing concurrency.
