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
