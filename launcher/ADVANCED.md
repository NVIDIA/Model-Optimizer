# Advanced Guide

## Architecture

### Shared Core

The launcher is built on a shared `core.py` module used by both:

- **`launch.py`** — public-facing launcher (this repo)
- **`slurm.py`** — internal CI orchestrator ([nmm-sandbox](https://gitlab-master.nvidia.com/omniml/integration/nmm-sandbox))

```text
core.py (shared)
├── Dataclasses: SandboxTask, SandboxPipeline, GlobalVariables
├── Executor builders: build_slurm_executor(), build_docker_executor()
├── Job runner: run_jobs()
├── Version reporter: report_versions()
├── Factory registry: register_factory(), set_slurm_config_type()
└── Default env: get_default_env()

launch.py                              slurm.py (nmm-sandbox)
├── imports core.py                    ├── imports core.py (via sys.path)
├── slurm_config.py (env-var driven)   ├── tools/slurm_config.py (cluster-specific)
├── registers: slurm_factory           ├── registers: oci_hsg, cw_dfw, computelab, ...
├── packager (LAUNCHER_DIR relative)   ├── packager (repo root relative)
└── launch() entrypoint                └── cicd() entrypoint
```

### Code Packaging

When a job is submitted, `PatternPackager` creates a tar.gz of the source code and rsyncs it to the cluster. The `code/` directory on the remote mirrors the launcher structure:

```text
code/
├── modules/
│   ├── Megatron-LM/megatron/...      # Training framework
│   └── Model-Optimizer/modelopt/...   # ModelOpt library (mounted over container install)
└── common/
    ├── megatron-lm/quantize/
    │   └── quantize.sh               # PTQ quantization + MMLU
    ├── tensorrt-llm/query.sh          # TRT-LLM server + query
    ├── vllm/query.sh                  # vLLM server + query
    ├── eagle3/                        # EAGLE3 pipeline scripts
    └── query.py                       # OpenAI-compatible query client
```

### ModelOpt Mount Mechanism

The container image (e.g., `nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc5`) ships with a pre-installed version of ModelOpt at a fixed path like `/usr/local/lib/python3.12/dist-packages/modelopt`. The launcher **bind-mounts your local `modelopt/` over this path**, so your local changes take effect without rebuilding the container.

The mount is configured via `modelopt_install_path` in `SlurmConfig`:

```yaml
slurm_config:
  modelopt_install_path: /usr/local/lib/python3.12/dist-packages/modelopt
```

At runtime, the executor constructs the mount:

- **Slurm**: `{job_dir}/{experiment_title}/{exp_id}/{task}/code/modules/Model-Optimizer/modelopt` → `{modelopt_install_path}`
- **Docker**: `{LAUNCHER_DIR}/modules/Model-Optimizer/modelopt` → `{modelopt_install_path}` (follows the symlink to the parent's `modelopt/`)

This means:

1. You can edit `modelopt/` source code locally
2. Submit a job — the packager tars your changes and ships them to the cluster
3. On the cluster, the container sees your modified `modelopt/` instead of the pre-installed one
4. No container rebuild needed for iterating on ModelOpt changes

The `modelopt_install_path` varies by container image. Check with:

```bash
docker run --rm <image> python3 -c "import modelopt; print(modelopt.__file__)"
```

### Model-Optimizer Symlink

`launcher/modules/Model-Optimizer` is a **symlink** to `../..` (the Model-Optimizer root), not a git submodule. This avoids recursive nesting — the launcher lives inside Model-Optimizer and references its own parent.

- Git tracks the symlink natively (`git clone` preserves it)
- `launch.py` auto-creates the symlink on first run if it's missing
- The packager's `find` follows symlinks, so `modules/Model-Optimizer/modelopt/*` resolves correctly

### Factory System

Slurm cluster configs use a factory pattern. YAMLs reference a factory by name:

```yaml
slurm_config:
  _factory_: "slurm_factory"
  nodes: 1
```

Factories are registered at import time via `register_factory()`. In `launch.py`, `slurm_factory` reads from environment variables (`SLURM_HOST`, `SLURM_ACCOUNT`, etc.). In `slurm.py`, `slurm_factory` resolves to a cluster-specific factory based on `SLURM_CLUSTER`:

```bash
# Default (OCI-HSG)
uv run slurm.py --yaml config.yaml --yes

# Switch cluster
SLURM_CLUSTER=cw_dfw uv run slurm.py --yaml config.yaml --yes
```

### YAML Formats

**`--yaml` format** (recommended) — maps top-level keys to function args:

```yaml
job_name: Qwen3-8B_NVFP4
pipeline:
  task_0:
    script: common/megatron-lm/quantize/quantize.sh
    slurm_config:
      _factory_: "slurm_factory"
```

**`pipeline=@` format** — bare pipeline without wrapper:

```yaml
task_0:
  script: common/megatron-lm/quantize/quantize.sh
  slurm_config:
    _factory_: "slurm_factory"
```

**Test YAML format** — list of jobs with `_target_` and overrides, used by nmm-sandbox's `tools/run_test_yaml.sh` for CI:

```yaml
- _target_: Qwen/Qwen3-8B/megatron_lm_ptq.yaml
  pipeline:
    allow_to_fail: true
    skip: false
    note: "known flaky"
```

Overrides are flattened to dot-notation and passed as nemo-run CLI args (e.g., `pipeline.allow_to_fail=True`).

### Global Variables

Pipeline YAMLs support `<<global_vars.X>>` interpolation for sharing values across tasks:

```yaml
pipeline:
  global_vars:
    hf_model: /hf-local/Qwen/Qwen3-8B

  task_0:
    environment:
      - HF_MODEL_CKPT: <<global_vars.hf_model>>

  task_1:
    environment:
      - HF_MODEL_CKPT: <<global_vars.hf_model>>
```

This is resolved in `SandboxPipeline.__post_init__` using regex substitution, not OmegaConf (which fails on isolated sub-configs in nemo-run).

### Metadata

Each experiment writes `metadata.json` to `experiments/<title>/<id>/`:

```json
{
  "experiment_id": "cicd_1773420387",
  "job_name": "Qwen3-8B_NVFP4_DEFAULT_CFG",
  "allow_to_fail": false,
  "note": ""
}
```

This is used by:

- `tools/wait_for_experiments.sh` — skip blocking on `allow_to_fail` failures
- `tools/post_review_to_gitlab.sh` — create/update GitLab issues for allowed failures
- Claude Code's `review-logs` skill — emit `<system-out>` instead of `<failure>` in JUnit XML

## Using Claude Code with the Launcher

Claude Code can create a tight feedback loop for model quantization experiments: configure → submit → monitor → diagnose → fix → resubmit — all from the CLI.

### Setup

Install Claude Code and ensure the launcher is ready:

```bash
npm install -g @anthropic-ai/claude-code
cd Model-Optimizer/launcher
git submodule update --init --recursive
```

### Workflow: Submit and Monitor

Ask Claude Code to launch a job and wait for results:

```text
> Run Qwen3-8B quantization on OCI-HSG and wait for it to finish

Claude will:
1. Run: uv run launch.py --yaml Qwen/Qwen3-8B/megatron_lm_ptq.yaml --yes
2. Monitor with: NEMORUN_HOME=$(pwd) uv run nemo experiment status <id>
3. Fetch logs when done: NEMORUN_HOME=$(pwd) uv run nemo experiment logs <id> 0
4. Report the MMLU score and pass/fail status
```

### Workflow: Diagnose Failures

When a job fails, ask Claude Code to analyze the logs:

```text
> /review-logs

Claude will:
1. Find all experiments in experiments/
2. Fetch logs via nemo experiment logs
3. Read and analyze error tracebacks
4. Produce a structured report with root cause and suggested fix
5. Write a JUnit XML for CI integration
```

### Workflow: Add a New Model

Ask Claude Code to set up a new model config:

```text
> Add Llama-3.1-70B quantization config. It needs 2 nodes with 4 GPUs each.

Claude will:
1. Create Meta/Llama-3.1-70B/megatron_lm_ptq.yaml
2. Set appropriate TP/EP based on model size
3. Reference the correct service script
4. Test with --dryrun to verify the config
```

### Workflow: Iterate on Failures

Claude Code can fix issues and resubmit in a loop:

```text
> The job failed with CUDA OOM. Try reducing the sequence length to 4096 and resubmit.

Claude will:
1. Edit the YAML config
2. Resubmit with uv run launch.py --yaml <config> --yes
3. Monitor and report results
```

### Workflow: Reproduce and Compare

Use `--to-yaml` to capture configs and compare runs:

```text
> Dump the resolved config for Qwen3-8B, then run it on both OCI-HSG and CW-DFW

Claude will:
1. Dump: uv run launch.py --yaml Qwen/Qwen3-8B/megatron_lm_ptq.yaml --to-yaml resolved.yaml
2. Run on OCI-HSG: SLURM_CLUSTER=oci_hsg uv run slurm.py --yaml resolved.yaml --yes
3. Run on CW-DFW: SLURM_CLUSTER=cw_dfw uv run slurm.py --yaml resolved.yaml --yes
4. Compare MMLU results
```

### Skills

The following Claude Code skills are available in the nmm-sandbox project:

| Skill | Trigger | Description |
|---|---|---|
| `/review-logs` | After job completion or failure | Analyze experiment logs, diagnose failures, produce JUnit XML |
| `/wait-for-jobs` | After detached submission | Poll experiment status until all jobs finish |
| `/eagle3-new-model` | Adding a new EAGLE3 model | Generate pipeline YAML for a new model |

### CI Integration

In CI, Claude Code runs automatically after each test job to:

1. Fetch and analyze all experiment logs
2. Generate `claude_analysis.md` with structured findings
3. Write `claude_review_rspec.xml` for GitLab test reporting
4. Post failure summaries as MR comments (via `tools/post_review_to_gitlab.sh`)
5. Create/update GitLab issues for `allow_to_fail` jobs that are consistently failing

If the main script crashes before the review runs, an `after_script` fallback posts the captured job output to the MR so failures are always visible.
