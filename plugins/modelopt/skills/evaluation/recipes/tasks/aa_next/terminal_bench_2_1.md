# Terminal-Bench 2.1 (AA) — nel-next / harbor

**Read `references/nel-next.md` first** — it covers the separate nel-next venv, the
`services`/`benchmarks`/`cluster`/`output` schema, AWS creds, the harbor/Fargate
architecture, `eval_image` arch, timeout strategy, MLflow export, the canary
syntax, and the run flow. This file is only the Terminal-Bench 2.1 deltas. Start
from `recipes/examples/example_eval_next.yaml`.

TB 2.1 is the successor to TB 2.0 — **identical harbor flow; only the playbook
name changes** (`terminal_bench_2_1` vs `terminal_bench_2`). The 2.1 task set is
pinned via a vendored registry override shipped in the `nemo-evaluator` package.

## Task-specific values

| Field | Value |
|---|---|
| `playbook` | `terminal_bench_2_1` (`harbor://terminal-bench@2.1`) |
| agent | `terminus-2` (from the playbook) |
| `repeats` | `8` (AA / leaderboard count — don't lower for a scored run) |
| sandbox | `ecs_fargate`, `stateful: true` (agent + verifier share one container) |
| `sandbox.region` | `${HARBOR_ECS_REGION:-us-east-1}` — one env var now drives region **and** repo name |
| `sandbox.ecr_repository` | `${HARBOR_ECR_REPOSITORY}` — set by `modelopttools:eval-config`; canonical value is `463701203462.dkr.ecr.<region>.amazonaws.com/harbor-<region>`, i.e. the repo **name tracks the region** (`harbor-us-east-1`). Changing region means changing both halves. |
| `cluster.container_env.AWS_DEFAULT_REGION` | match `sandbox.region` |
| `sandbox.log_stream_prefix` | canonical stem `terminalbench-21`; the reviewed leaf appends model + framework (`terminalbench-21-nano-v35-vllm`) |
| `max_concurrent` / `sandbox.concurrency` | `50` (canonical bench.yaml) |
| timeout_strategy | `max` (canonical bench.yaml) + `agent_kwargs.llm_kwargs.timeout: 3600`; use `task` for leaderboard-comparable |
| `cluster.eval_image` | **`0.5.0.1-harbor`** (`${NEL_NEXT_EVAL_IMAGE}`, multi-arch) *(shared — see `references/nel-next.md`)* |
| `proxy.request_timeout` | `3600` — must be **≥** `agent_kwargs.llm_kwargs.timeout` *(shared — see `references/nel-next.md`)* |
| `drop_params` | `max_tokens`, `max_completion_tokens`, `max_input_tokens_per_task`, `no_rebuild` *(shared — see `references/nel-next.md`)* |
| `output.export_config.mlflow.exclude_patterns` | `["shard*", "model_traffic.jsonl"]` *(shared — see `references/nel-next.md`)* |
| `http_pairs_dump` | **first** in the interceptor chain (same position as SWE-bench) — canary/diagnostic only, drop it for a scored run (unbounded error-pair retention) |
| `proxy.model_traffic.capture_request_body` | `true`, set **per service** (`services.<alias>.proxy.model_traffic`) — never in a shared block, which would inject an alias-only entry with no `type` into other models' composed configs |
| `output.export_config.mlflow.tags` | canonical adds `task_name: terminal-bench-2.1` and `nemo-evaluator-next-version: 0.5.0.1` alongside `model`/`framework` |
| scope | 89 tasks × `repeats: 8` |

These values mirror the canonical TB2.1 config — re-check it before a scored run:
`configs/benchmarks/terminal-bench-2.1/bench.yaml` (+ `manifest.yaml`) in
nvidia-eval-factory-benchmarking (`dl/JoC/competitive_evaluation/…`), with the image pin in
`configs/shared/nel_next_containers.yaml`. See `references/nel-next.md` + the eval-config
"source of truth" note.

> **`bench_direct.yaml` is a different backend, not an alternative spelling.** The same
> upstream dir now also ships a Gym-native route (Terminus 2 agent + opensandbox,
> `++num_samples_in_parallel=512`, its own Gym commit pin) used by the P/D recipes. It
> scores the same 2.1 task set but through different machinery; `benchmark_backend:
> evaluator` — the harbor path below — is what this skill runs and what the nel-next
> deltas here describe. Don't mix values between the two.

The `benchmarks:` block (drop into the example template):

```yaml
benchmarks:
  - playbook: terminal_bench_2_1
    repeats: 8
    max_concurrent: 50            # canonical; keep == sandbox.concurrency
    solver:
      service: <svc-name>
      timeout_strategy: max       # canonical bench.yaml; use "task" for leaderboard-comparable
      run_timeout: 7200           # per-task agent wall-clock ceiling (2h)
      agent_kwargs:
        llm_kwargs:
          timeout: 3600           # per-request LLM timeout (canonical)
    sandbox:
      region: ${HARBOR_ECS_REGION:-us-east-1}  # canonical; repo name below must track it
      ecr_repository: ${HARBOR_ECR_REPOSITORY} # from eval-config (internal harbor account/region)
      concurrency: 50
      log_stream_prefix: terminalbench-21-<model>-<framework>
```

`cluster.eval_image: ${NEL_NEXT_EVAL_IMAGE}` (`0.5.0.1-harbor`) and the AWS creds
come from `modelopttools:eval-config` (run it first) + the workspace `.env`.

**Sharding.** `max_concurrent`/`sandbox.concurrency` are **per shard**, and each shard runs
its own vLLM on its own node — `shards: N` multiplies both serving capacity and live Fargate
sandboxes (`N × concurrency`). Trials are partitioned and merged, so the score is unaffected;
it is purely a wall-clock lever. `shards: 4` suits 89 × r8 = 712 trials. Check
`N × concurrency` against the Fargate quota and `N × gpus_per_node` against your allocation.

## Score Extraction

Report **`pass@1`** — the key the upstream `manifest.yaml` certifies on, and the only
one to quote. Benchmark `terminal-bench@2.1`, scorer `pass@1` (0–1):
the resolved rate over the 2.1 task set, **already averaged over repeats** (a single
`pass@1`; no `avg-of-N` key). MLflow logs it as `pass_at_1`. Read from `report.md`
(Benchmark / Scorer table) or `nel eval report -r <run_id>`, then push to MLflow with
`nel-next.sh mlflow-push -r <run_id> -c <cfg>` (SLURM doesn't auto-export). Keep
`timeout_strategy` fixed across baseline vs quantized for a valid delta. (Terminal-Bench
2.0 and 2.1 use different task sets, so their `pass@1` numbers aren't directly comparable.)
