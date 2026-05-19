# Debug failed runs

Copy this checklist and track your progress:

```
Debug progress:
- [ ] Step 1: Gather from the user
- [ ] Step 2: Get job info
- [ ] Step 3: Copy and check logs
- [ ] Step 4: Apply fix
- [ ] Step 5: Verify fix
```

## Step 1: Gather from the user

- **Invocation ID**: The failed run to debug.
- **Error symptoms** (optional): What the user observed (timeout, OOM, etc.).

## Step 2: Get job info

```bash
uv run nemo-evaluator-launcher status <invocation_id> --json
uv run nemo-evaluator-launcher info <invocation_id>
```

Extract from output:

- **Status**: Job state per task
- **Logs path**: Remote path to logs directory
- **Slurm Job ID**: Job ID for log filenames
- **Hostname**: Cluster login node for SSH

## Step 3: Copy and check logs

**IMPORTANT**: Copy what you need (and only what you need) locally BEFORE analysis — each SSH command requires user approval, so remote one-by-one reads are disruptive, and copying too much is slow.

```bash
uv run nemo-evaluator-launcher info <invocation_id> --copy-logs /tmp/debug-logs
```

```bash
LOGS=/tmp/debug-logs/<job_id>/logs

# Check logs in order:
# 1. slurm log - job-level errors (scheduling, walltime, preemption)
cat $LOGS/slurm-*.log

# 2. server log - deployment errors (OOM, missing model, bad args, driver mismatch)
tail -200 $LOGS/server-*-0.log
grep -i -E '(error|exception|failed|OOM|killed)' $LOGS/server-*-0.log | tail -50

# 3. proxy log - load balancer errors (multi-instance only)
cat $LOGS/proxy-*.log 2>/dev/null

# 4. client log - evaluation errors (dataset, scorer, timeout, rate limiting)
tail -200 $LOGS/client-*.log
```

- **slurm-*.log** — Job-level errors (health check timeouts, account/partition errors, walltime exceeded, preemption)
- **server-*-N.log** — Deployment errors (CUDA OOM, missing model/checkpoint, bad extra_args, GPU driver mismatch, image pull failure)
- **proxy-*.log** — HAProxy load balancer errors (only present with multi-instance deployments)
- **client-*.log** — Evaluation errors (dataset access, scorer errors, timeouts, rate limiting)

**IMPORTANT**: Always check BOTH server AND client logs. Client logs show symptoms (e.g., `unknown_agent_error`, `failed_samples_policy`), server logs show actual cause.

## Step 4: Apply fix

**Common fixes:**

- **CUDA OOM**: Increase `deployment.tensor_parallel_size` to shard across more GPUs. For multi-node: increase `execution.num_nodes` and set `deployment.pipeline_parallel_size`. As last resort: add `--max-model-len <lower_value>` to `deployment.extra_args`. Do NOT quantize as a first fix — scale compute instead.
- **Missing model/checkpoint**: `FileNotFoundError` or `RepositoryNotFoundError` or `GatedRepoError: 403` — verify `deployment.checkpoint_path` or `deployment.hf_model_handle`. For gated models, set `HF_TOKEN` via `deployment.env_vars`.
- **Bad `extra_args`**: `unrecognized arguments` or `unexpected keyword argument` — check flags against deployment engine version. Some flags change between versions (e.g., `--rope-scaling` removed in vLLM > 0.11.0).
- **Image pull failure**: `manifest not found` or `pyxis: child 1 failed` — verify image tag exists. Drop `:5005` from GitLab container registry URLs.
- **GPU driver mismatch**: `CUDA driver version is insufficient` — use an older container image matching the host CUDA driver.
- **Health check timeout / connection refused**: Server didn't start — check server logs first. Increase `execution.endpoint_readiness_timeout` (seconds). SLURM default: `null` (falls back to walltime).
- **Server crashed mid-eval**: `Connection reset by peer` — check server logs for OOM. Reduce `parallelism` (concurrent requests). Check SLURM logs for preemption or walltime exceeded.
- **Missing dataset**: `DatasetNotFoundError` or `GatedRepoError: 403` — accept the license on HuggingFace, set `HF_TOKEN` in `evaluation[].env_vars`.
- **Scorer errors**: `ScorerError` or `KeyError` — check model output format, `adapter_config`, and `max_new_tokens`.
- **Timeout**: `TimeoutError` or `Request timed out` — increase `evaluation[].nemo_evaluator_config.config.params.request_timeout`. Reduce `max_new_tokens` or `parallelism` if overloaded.
- **Config validation**: `MissingMandatoryValue` (unfilled `???`), `ValidationError` (type mismatch), `ScannerError` (invalid YAML) — run `--dry-run` to catch these upfront.
- **Walltime exceeded**: `CANCELLED DUE TO TIME LIMIT` — NEL submits paired restart jobs that automatically resume when walltime expires, so this is often expected behavior, not a failure. Only increase `execution.walltime` if the evaluation isn't making progress across restarts.
- **Preemption**: `CANCELLED DUE TO PREEMPTION` — the paired restart job should automatically resume. If it doesn't, use non-preemptible partition, or re-run.
- **Container not found**: Applies to both `deployment.image` and task-level eval container. Drop `:5005` from GitLab registry URLs.
- Troubleshooting docs: list files with WebFetch `https://api.github.com/repos/NVIDIA-NeMo/Evaluator/contents/docs/troubleshooting`, then fetch relevant ones from `https://raw.githubusercontent.com/NVIDIA-NeMo/Evaluator/main/docs/troubleshooting/<file>`

**Fix Slurm invalid account/partition:**

```bash
# Get cluster hostname from nel info
uv run nemo-evaluator-launcher info <invocation_id>

# Check available accounts on the cluster
ssh <user>@<hostname> "sacctmgr show user <user> withassoc format=Account%30,Partition%20 --noheader"
```

**Fix HuggingFace API 429 Rate Limiting:**

Always set `HF_TOKEN` in both `deployment.env_vars` and `evaluation[].env_vars`, even for public models. To pre-cache:

```bash
ssh <user>@<cluster-hostname>
python3 -m venv .venv && source .venv/bin/activate && pip install -U huggingface_hub
export HF_HOME=<your-cache-path>/huggingface
export HF_TOKEN=<your-token>
huggingface-cli download <org>/<model>
```

Then set `HF_HUB_OFFLINE: 1` in config's env_vars.

**Correctness warning** — these fixes affect evaluation results:
- `--max-model-len` — restricts context window, may truncate prompts
- `temperature` — sampling randomness
- `top_p` — nucleus sampling threshold
- `max_new_tokens` — output truncation if too low

## Step 5: Verify fix

```bash
# 1. Dry-run (validates config without running)
uv run nemo-evaluator-launcher run --config <config> --dry-run

# 2. Smoke test (10 samples)
uv run nemo-evaluator-launcher run --config <config> -o ++evaluation.nemo_evaluator_config.config.params.limit_samples=10

# 3. Single failing task only
uv run nemo-evaluator-launcher run --config <config> -t <failed_task> -o ++evaluation.nemo_evaluator_config.config.params.limit_samples=10

# 4. Monitor
uv run nemo-evaluator-launcher status <new_invocation_id> --json
```
