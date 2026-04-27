---
name: launching-evals
description: Run, monitor, analyze, and debug LLM evaluations via nemo-evaluator-launcher. Covers running evaluations, checking status and live progress, debugging failed runs, exporting artifacts and logs, and analyzing results. ALWAYS triggers on mentions of running evaluations, checking progress, debugging failed evals, analyzing or analysing runs or results, run directories or artifact paths on clusters, Slurm job issues, invocation IDs, or inspecting logs (client logs, server logs, SSH to cluster, tail logs, grep logs). Do NOT use for creating or modifying evaluation configs.
license: Apache-2.0
# Vendored verbatim from NVIDIA NeMo Evaluator (commit 8fa16b2)
# https://github.com/NVIDIA-NeMo/Evaluator/tree/8fa16b237d11e213ea665d5bad6b44d393762317/packages/nemo-evaluator-launcher/.claude/skills/launching-evals
# To re-sync: .claude/scripts/sync-upstream-skills.sh
---

# NeMo Evaluator Skill

## Quick Reference

### nemo-evaluator-launcher CLI

```bash
# Run evaluation
uv run nemo-evaluator-launcher run --config <path.yaml>
uv run nemo-evaluator-launcher run --config <path.yaml> -t <a_single_task_to_be_run_by_name>
uv run nemo-evaluator-launcher run --config <path.yaml> -t <task_name_1> -t <task_name_2> ...
uv run nemo-evaluator-launcher run --config <path.yaml> -o evaluation.nemo_evaluator_config.config.params.limit_samples=10 ...

# Preview the resolved config and the sbatch script without running the evaluation
uv run nemo-evaluator-launcher run --config <path.yaml> --dry-run

# Check status (--json for machine-readable output)
uv run nemo-evaluator-launcher status <invocation_id> --json

# Get evaluation run info (output paths, slurm job IDs, cluster hostname, etc.)
uv run nemo-evaluator-launcher info <invocation_id>

# Copy just the logs (quick — good for debugging)
uv run nemo-evaluator-launcher info <invocation_id> --copy-logs ./evaluation-results/

# For artifacts: use `nel info` to discover paths. If remote, SSH to explore and rsync what you need.
# If local, just read directly from the paths shown by `nel info`.
# ssh <user>@<hostname> "ls <artifacts_path>/"
# rsync -avzP <user>@<hostname>:<artifacts_path>/{results.yml,eval_factory_metrics.json,config.yml} ./evaluation-results/<invocation_id>.<job_index>/artifacts/

# Resume a failed/interrupted run (re-sbatches existing run.sub in the original run directory)
uv run nemo-evaluator-launcher resume <invocation_id>

# List past runs
uv run nemo-evaluator-launcher ls runs --since 1d   

# List available evaluation tasks (by default, only shows tasks from the latest released containers)
uv run nemo-evaluator-launcher ls tasks
uv run nemo-evaluator-launcher ls tasks --from_container nvcr.io/nvidia/eval-factory/simple-evals:26.03
```

## Workflow

The complete evaluation workflow is divided into the following steps you should follow IN ORDER.

1. Create or modify a config using the `nel-assistant` skill. If the user provides a past run, use its `config.yml` artifact as a starting point.
2. Run the evaluation. See `references/run-evaluation.md` when executing this step.
3. **Monitor progress (MANDATORY after every `nel run`)**: poll status repeatedly until SUCCESS/FAILED. See `references/check-progress.md`.
4. Post-run actions (when terminal state reached):
   1. When the evaluation status is `SUCCESS`, analyze the results. See `references/analyze-results.md` when executing this step.
   2. When the evaluation status is `FAILED`, debug the failed run. See `references/debug-failed-runs.md` when executing this step.

# Key Facts

- Benchmark-specific info learned during launching/analyzing evals should be added to `references/benchmarks/`
- **PPP** = Slurm account (the `account` field in cluster_config.yaml). When the user says "change PPP to X", update the account value (e.g., `coreai_dlalgo_compeval` → `coreai_dlalgo_llm`).
- **Slurm job pairs**: NEL (nemo-evaluator-launcher) submits paired Slurm jobs — a RUNNING job + a PENDING restart job (for when the 4h walltime expires). Never cancel the pending restart jobs — they are expected and necessary.
- **HF cache requirement**: For configs with `HF_HUB_OFFLINE=1`, models must be pre-downloaded to the HF cache on each cluster before launching. **Before running a model on a new cluster, always ask the user if the model is already cached there.** If not, on the cluster login node: `python3 -m venv hf_cli && source hf_cli/bin/activate && pip install huggingface_hub` then `HF_HOME=/lustre/fsw/portfolios/coreai/users/<username>/cache/huggingface hf download <model>`. Without this, vLLM will fail with `LocalEntryNotFoundError`.
- **`data_parallel_size` is per node**: `dp_size=1` with `num_nodes=8` means 8 model instances total (one per node), load-balanced by haproxy. Do NOT interpret `dp_size` as the global replica count.
- **`payload_modifier` interceptor**: The `params_to_remove` list (e.g. `[max_tokens, max_completion_tokens]`) strips those fields from the outgoing payload, intentionally lifting output length limits so reasoning models can think as long as they need.
- **Auto-export git workaround**: The export container (`python:3.12-slim`) lacks `git`. When installing the launcher from a git URL, set `auto_export.launcher_install_cmd` to install git first (e.g., `apt-get update -qq && apt-get install -qq -y git && pip install "nemo-evaluator-launcher[all] @ git+...#subdirectory=packages/nemo-evaluator-launcher"`).
- **Do NOT use `nemo-evaluator-launcher export --dest local`** — it only writes a summary JSON (`processed_results.json`), it does NOT copy actual logs or artifacts despite accepting `--copy_logs` and `--copy-artifacts` flags. `nel info --copy-artifacts` works but copies everything (very slow for large benchmarks). Preferred approach: use `nel info` to discover paths — if local, read directly; if remote, SSH to explore and rsync only what you need. Note that `nel info` prints standard artifacts but benchmarks produce additional artifacts in subdirs — explore to find them.
