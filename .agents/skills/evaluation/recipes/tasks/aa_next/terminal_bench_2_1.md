# Terminal-Bench 2.1 (AA) — nel-next / harbor

**Read `references/nel-next.md` first** — it covers the separate 0.3.x venv, the
`services`/`benchmarks`/`cluster`/`output` schema, AWS creds, the harbor/Fargate
architecture, `eval_image` arch, timeout strategy, MLflow export, the canary
syntax, and the run flow. This file is only the Terminal-Bench 2.1 deltas. Start
from `recipes/examples/example_eval_next.yaml`.

TB 2.1 is the successor to TB 2.0 — **identical harbor flow; only the playbook
name changes** (`terminal_bench_2_1` vs `terminal_bench_2`). The 2.1 task set is
pinned via a vendored registry override in nemo-evaluator-next.

## Task-specific values

| Field | Value |
|---|---|
| `playbook` | `terminal_bench_2_1` (`harbor://terminal-bench@2.1`) |
| agent | `terminus-2` (from the playbook) |
| `repeats` | `8` (AA / leaderboard count — don't lower for a scored run) |
| sandbox | `ecs_fargate`, `stateful: true` (agent + verifier share one container) |
| `sandbox.region` | match the region in `${HARBOR_ECR_REPOSITORY}` (us-east-1 with the eval-config default) |
| `sandbox.ecr_repository` | `${HARBOR_ECR_REPOSITORY}` — set by the `modelopttools:eval-config` skill (internal harbor ECR) |
| `cluster.eval_image` | `${NEL_NEXT_EVAL_IMAGE}` — set by `modelopttools:eval-config` |
| `cluster.container_env.AWS_DEFAULT_REGION` | match `sandbox.region` |
| `max_concurrent` / `sandbox.concurrency` | `50` (canonical bench.yaml) |
| timeout_strategy | `max` (canonical bench.yaml) + `agent_kwargs.llm_kwargs.timeout: 3600`; use `task` for leaderboard-comparable |
| `cluster.eval_image` requirement | **≥ `0.3.1.1-harbor`** — TB 2.1's task set is pinned via a vendored registry override in that image (`${NEL_NEXT_EVAL_IMAGE}`, multi-arch) |

These values mirror the canonical TB2.1 config — re-check it before a scored run:
`configs/benchmarks/nel_next/terminal_bench_21/bench.yaml` in
nvidia-eval-factory-benchmarking (see `references/nel-next.md` + the eval-config
"source of truth" note). The `benchmarks:` block (drop into the example template):

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
      region: us-east-1                       # must match the region in ${HARBOR_ECR_REPOSITORY}
      ecr_repository: ${HARBOR_ECR_REPOSITORY} # from eval-config (internal harbor account/region)
      concurrency: 50
      log_stream_prefix: terminalbench21-<model>-<cluster>
```

`cluster.eval_image: ${NEL_NEXT_EVAL_IMAGE}` (≥ `0.3.1.1-harbor`) and the AWS creds
come from `modelopttools:eval-config` (run it first) + the workspace `.env`.

## Score Extraction

Report **`pass@1`** only — benchmark `terminal-bench@2.1`, scorer `pass@1` (0–1):
the resolved rate over the 2.1 task set, **already averaged over repeats** (a single
`pass@1`; no `avg-of-N` key). MLflow logs it as `pass_at_1`. Read from `report.md`
(Benchmark / Scorer table) or `nel eval report -r <run_id>`, then push to MLflow with
`nel-next.sh mlflow-push -r <run_id> -c <cfg>` (SLURM doesn't auto-export). Keep
`timeout_strategy` fixed across baseline vs quantized for a valid delta. (Terminal-Bench
2.0 and 2.1 use different task sets, so their `pass@1` numbers aren't directly comparable.)
