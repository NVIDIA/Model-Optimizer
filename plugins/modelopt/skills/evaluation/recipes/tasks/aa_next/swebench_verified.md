# SWE-bench Verified (AA) — nel-next / harbor

**Read `references/nel-next.md` first** (shared venv/schema/AWS/architecture/MLflow/
run flow). Same harbor/ECS-Fargate flow as Terminal-Bench; the deltas are the
**OpenHands agent**, a larger problem set, longer timeouts, and a different
ECR/region. Start from `recipes/examples/example_eval_next.yaml`.

> **Source of truth:** `configs/benchmarks/swe-bench-verified/bench.yaml` in
> nvidia-eval-factory-benchmarking (`dl/JoC/competitive_evaluation/…`), with the eval-image
> pin in `configs/shared/nel_next_containers.yaml` — match its values for a reference run.

## Task-specific values (canonical `bench.yaml`)

| Field | Value |
|---|---|
| `playbook` | `swebench_verified` (`harbor://swebench-verified@1.0`) |
| agent | `openhands-sdk` (playbook; `agent_kwargs: {max_iterations: 200, version: "1.17.0"}`) |
| scope | 500 Python tasks × `repeats: 5` |
| `max_concurrent` / `sandbox.concurrency` | `15` in `bench.yaml`; per-model configs override it (MiniMax-M2.7 uses `20`) |
| `solver` | `timeout_strategy: max`, `run_timeout: 10800` (3h), `agent_kwargs.llm_kwargs.timeout: 3600` |
| `sandbox.region` | `us-east-2` |
| `sandbox.ecr_repository` | `${HARBOR_SWEBENCH_ECR_REPOSITORY}` (dedicated `harbor-swebench` repo, **us-west-2**, regardless of sandbox region) |
| `cluster.eval_image` | `${NEL_NEXT_EVAL_IMAGE}` → **`0.5.0.1-harbor`** (same pin as TB2.1: `configs/shared/nel_next_containers.yaml`) *(shared — see `references/nel-next.md`)* |
| `cluster.container_env.AWS_DEFAULT_REGION` | `us-east-2` (match `sandbox.region`) |
| `instruction_template` | `/configs/prompts/swebench_instruction.md`, **must be MOUNTED**; content is scoring-relevant (gotcha below) |
| `proxy.request_timeout` | `3600` (FEP-1104 paired HTTP timeout; leaves mirror it on the service proxy) *(shared — see `references/nel-next.md`)* |
| `drop_params` | `max_tokens`, `max_completion_tokens`, `max_input_tokens_per_task`, `no_rebuild` *(shared — see `references/nel-next.md`)* |
| `output.export_config.mlflow.exclude_patterns` | `["shard*", "model_traffic.jsonl"]` *(shared — see `references/nel-next.md`)* |
| `system_message` | `strategy: replace` + the OpenHands prompt from `bench.yaml` (verbatim) — scoring-relevant |

```yaml
benchmarks:
  - playbook: swebench_verified
    repeats: 5
    max_concurrent: 15            # keep == sandbox.concurrency; per-model configs may raise both
    instruction_template: /configs/prompts/swebench_instruction.md   # mounted (see gotcha)
    solver:
      service: <svc-name>
      timeout_strategy: max          # canonical; "task" = leaderboard-comparable
      run_timeout: 10800
      agent_kwargs: {llm_kwargs: {timeout: 3600}}
    sandbox:
      region: us-east-2
      ecr_repository: ${HARBOR_SWEBENCH_ECR_REPOSITORY}
      concurrency: 15
      log_stream_prefix: swebench-verified-<model>-<cluster>
```

### Gotcha — mount the instruction template

The playbook defaults `instruction_template: swebench-instruction.md`, but the
harbor image doesn't ship that built-in → run dies at finalize with
`FileNotFoundError: instruction_template not found`. So it must be mounted.

**Which file you mount changes the score.** Mount the canonical `swebench_instruction.md`
(underscore) at `/configs/prompts/swebench_instruction.md`, taken from the reference config or
run dir. The built-in in the `nemo_evaluator/templates/` venv directory is a **different
prompt** (`swebench-instruction.md`, hyphen) — it runs, but results are not comparable. Keep
whichever you use fixed across both sides of a comparison.

**Both sides of a BF16-vs-quantized comparison must mount the same file**, or the delta
includes a prompt change. The prompt is internal — there is no public download. Two ways to
obtain it:

```bash
# (a) From a reference run dir — the exact file a scored run mounted. Read that run's
#     full_config.yaml `cluster.container_mounts` entry for swebench_instruction.md, then:
scp <ref-cluster>:<path-from-that-config> ./swebench_instruction.md
# (b) From the eval-factory repo — the path `_swebench_verified.instruction_template` points at
#     in configs/benchmarks/swe-bench-verified/bench.yaml
#     (dl/JoC/competitive_evaluation/nvidia-eval-factory-benchmarking).

sha256sum ./swebench_instruction.md          # record this next to the score
ssh <login> 'mkdir -p <lustre>/<user>/prompts'
scp ./swebench_instruction.md <login>:<lustre>/<user>/prompts/
ssh <login> 'sha256sum <lustre>/<user>/prompts/swebench_instruction.md'   # must match the line above
```

```yaml
benchmarks: [{playbook: swebench_verified, instruction_template: /configs/prompts/swebench_instruction.md}]
cluster:
  container_mounts: ["<lustre>/<user>/prompts/swebench_instruction.md:/configs/prompts/swebench_instruction.md:ro"]
```

### Deployment proxy (multi-turn agentic)

OpenHands runs ~200 turns/task. The canonical config adds a `system_message`
interceptor (a large OpenHands system prompt — copy it verbatim from `bench.yaml`)
plus `turn_counter`.

**Order differs from TB2.1**: `http_pairs_dump` is **first** (not last) and `drop_params`
comes **before** `consolidate_system`. `http_pairs_dump` is canary/diagnostic-only — it
retains every error pair in memory for the whole run (`references/nel-next.md`); drop it
from the scored config.

```yaml
proxy:
  request_timeout: 3600
  extra_body: {skip_special_tokens: false}   # add model-card sampling extras if the card sets them
  model_traffic: {capture_request_body: true}   # FEA-224; adds the upstream request body to the traffic capture that is ALREADY ON by default
  interceptors:
    # - {name: http_pairs_dump, config: {dump_path: "$${NEL_OUTPUT_DIR}/http_pairs_metrics.json", first_n: 50}}   # canary only
    - {name: system_message, config: {strategy: replace, system_message: "<the OpenHands prompt from bench.yaml>"}}
    - {name: turn_counter, config: {max_turns: 200, position: system_message}}
    - {name: drop_params, config: {params: [max_tokens, max_completion_tokens, max_input_tokens_per_task, no_rebuild]}}
    - {name: consolidate_system}
    - {name: reasoning}          # reasoning models: normalize reasoning field …
    - {name: reasoning_replay}   # … and replay across turns. Drop both for instruct models.
```

**`reasoning_replay.mode` is per model, not per benchmark.** Valid values are
`think_tags` / `native` / `both`, and the **default is `think_tags`** — leaving `mode` unset is
not a third behaviour, it selects `think_tags`. Qwen-style: `think_tags`. GLM: `native`.
MiniMax: leave unset (i.e. `think_tags`). Copying another model's mode is a silent
output-parsing bug.

**Omitting `system_message` is a scoring change**: without it the agent runs the
openhands-sdk default prompt instead of the canonical one.

### Sharding

`max_concurrent`/`sandbox.concurrency` are **per shard**, and each shard redeploys the model
on its own node — `shards: N` multiplies serving capacity *and* live Fargate sandboxes
(`N × concurrency`). 500 tasks × `repeats: 5` = 2500 trials; `shards: 10` at `concurrency: 15`
suits it. Score is unaffected — purely a wall-clock lever. Check `N × concurrency` against the
Fargate quota and `N × gpus_per_node` against your allocation.

## Score Extraction

Report **`pass@1`** only — benchmark `swebench-verified@1.0`, scorer `pass@1` (0–1):
the resolved rate over the 500 tasks, **already averaged over repeats** (nel-next
reports a single `pass@1`; there is **no `avg-of-N` key** like the 0.2.6 nemo-skills
metrics). MLflow logs it as `pass_at_1`. Read from `report.md` (Benchmark / Scorer
table) in the run dir or `nel eval report -r <run_id>`, then push to MLflow with
`nel-next.sh mlflow-push -r <run_id> -c <cfg>` (SLURM doesn't auto-export). Keep
`timeout_strategy` + the instruction/system prompt fixed across baseline vs quantized
for a valid delta.
