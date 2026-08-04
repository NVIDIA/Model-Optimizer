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
| `max_concurrent` / `sandbox.concurrency` | `15` in `bench.yaml`; **per-model leaves override it** (e.g. MiniMax-M2.7 uses `20`) — check the model's own golden leaf |
| `solver` | `timeout_strategy: max`, `run_timeout: 10800` (3h), `agent_kwargs.llm_kwargs.timeout: 3600` |
| `sandbox.region` | `us-east-2` |
| `sandbox.ecr_repository` | `${HARBOR_SWEBENCH_ECR_REPOSITORY}` (dedicated `harbor-swebench` repo, **us-west-2**, regardless of sandbox region) |
| `cluster.eval_image` | `${NEL_NEXT_EVAL_IMAGE}` — golden pin **`0.5.0.1-harbor`** (same single source as TB2.1: `configs/shared/nel_next_containers.yaml`). `0.3.1.1-harbor` is the bare minimum (FEP-1085 reasoning fix) but is two minors behind |
| `cluster.container_env.AWS_DEFAULT_REGION` | `us-east-2` (match `sandbox.region`) |
| `instruction_template` | `/configs/prompts/swebench_instruction.md`, **must be MOUNTED** — the harbor image doesn't bundle a built-in, and the template **content is scoring-relevant** (gotcha below) |
| `proxy.request_timeout` | `3600` (FEP-1104 paired HTTP timeout; leaves mirror it on the service proxy) |
| `drop_params` | `max_tokens`, `max_completion_tokens`, `max_input_tokens_per_task`, `no_rebuild` |
| `output.export_config.mlflow.exclude_patterns` | `["shard*", "model_traffic.jsonl"]` |
| `system_message` | `strategy: replace` + the canonical OpenHands prompt — **scoring-relevant**, copy verbatim from `bench.yaml` |

```yaml
benchmarks:
  - playbook: swebench_verified
    repeats: 5
    max_concurrent: 15
    instruction_template: /configs/swebench-instruction.md   # mounted (see gotcha)
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

**Which file you mount changes the score.** The canonical runs mount the internal compeval
prompt (`swebench_instruction.md` — note the underscore) at
`/configs/prompts/swebench_instruction.md`; take it from the canonical `bench.yaml` /
reference run dir, not from this repo. The public built-in shipped in the
`nemo_evaluator/templates/` venv directory is a **different prompt** (`swebench-instruction.md`,
hyphen) — it will run, but the result is not comparable to golden. Use the canonical one for
any scored or baseline-vs-quantized run, and keep it fixed across both sides of a comparison.

```bash
VENV="${NEL_NEXT_VENV:-$HOME/.local/share/nel/venvs/nel-next}"   # same default as nel-next.sh (NEL_NEXT_VENV may be unset)
cp "$VENV/lib/python3.12/site-packages/nemo_evaluator/templates/swebench-instruction.md" /tmp/
ssh <login> 'mkdir -p <lustre>/<user>/prompts' && scp /tmp/swebench-instruction.md <login>:<lustre>/<user>/prompts/
```

```yaml
benchmarks: [{playbook: swebench_verified, instruction_template: /configs/swebench-instruction.md}]
cluster:
  container_mounts: ["<lustre>/<user>/prompts/swebench-instruction.md:/configs/swebench-instruction.md:ro"]
```

### Deployment proxy (multi-turn agentic)

OpenHands runs ~200 turns/task. The canonical config adds a `system_message`
interceptor (a large OpenHands system prompt — copy it verbatim from `bench.yaml`)
plus `turn_counter`.

**Order matters and differs from TB2.1.** Here `http_pairs_dump` is **first** (not last)
and `drop_params` comes **before** `consolidate_system`. This is the canonical order,
verified against a reference oci-hsg run:

```yaml
proxy:
  request_timeout: 3600
  extra_body: {skip_special_tokens: false}   # add model-card sampling extras if the card sets them
  model_traffic: {capture_request_body: true}   # FEA-224; pair with the exclude_patterns entry
  interceptors:
    - {name: http_pairs_dump, config: {dump_path: "$${NEL_OUTPUT_DIR}/http_pairs_metrics.json", first_n: 50}}
    - {name: system_message, config: {strategy: replace, system_message: "<the OpenHands prompt from bench.yaml>"}}
    - {name: turn_counter, config: {max_turns: 200, position: system_message}}
    - {name: drop_params, config: {params: [max_tokens, max_completion_tokens, max_input_tokens_per_task, no_rebuild]}}
    - {name: consolidate_system}
    - {name: reasoning}          # reasoning models: normalize reasoning field …
    - {name: reasoning_replay}   # … and replay across turns. Drop both for instruct models.
```

**`reasoning_replay.mode` is per model, not per benchmark — take it from that model's own
golden leaf, never from another model's.** Known values: `think_tags` (Qwen-style),
`native` (GLM), and **omitted entirely** (MiniMax — the default is correct). Copying another
model's mode is a silent output-parsing bug, not a config nit.

**Omitting `system_message` is a scoring change, not a simplification** — without it the
agent runs on the openhands-sdk default prompt instead of the canonical one, and results
are not comparable to golden or to other models evaluated with it.

### Sharding

`max_concurrent`/`sandbox.concurrency` are **per shard**, and each shard redeploys the model
on its own node — so `shards: N` multiplies serving capacity *and* live Fargate sandboxes
(`N × concurrency`). SWE-bench is the heaviest AA benchmark here: 500 tasks × `repeats: 5`
= 2500 trials, and the reference run uses `shards: 10` (~250 trials/shard) at
`concurrency: 15`. Trials are partitioned and merged, so the score is unaffected — it is
purely a wall-clock lever. Check `N × concurrency` against the harbor Fargate quota and
`N × gpus_per_node` against your allocation before raising it.

## Score Extraction

Report **`pass@1`** only — benchmark `swebench-verified@1.0`, scorer `pass@1` (0–1):
the resolved rate over the 500 tasks, **already averaged over repeats** (nel-next
reports a single `pass@1`; there is **no `avg-of-N` key** like the 0.2.6 nemo-skills
metrics). MLflow logs it as `pass_at_1`. Read from `report.md` (Benchmark / Scorer
table) in the run dir or `nel eval report -r <run_id>`, then push to MLflow with
`nel-next.sh mlflow-push -r <run_id> -c <cfg>` (SLURM doesn't auto-export). Keep
`timeout_strategy` + the instruction/system prompt fixed across baseline vs quantized
for a valid delta.
