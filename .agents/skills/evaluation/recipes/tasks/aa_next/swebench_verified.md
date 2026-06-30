# SWE-bench Verified (AA) — nel-next / harbor

**Read `references/nel-next.md` first** (shared venv/schema/AWS/architecture/MLflow/
run flow). Same harbor/ECS-Fargate flow as Terminal-Bench; the deltas are the
**OpenHands agent**, a larger problem set, longer timeouts, and a different
ECR/region. Start from `recipes/examples/example_eval_next.yaml`.

> **Source of truth:** `configs/benchmarks/nel_next/swebench_verified/bench.yaml`
> in nvidia-eval-factory-benchmarking — match its values for a reference run.

## Task-specific values (canonical `bench.yaml`)

| Field | Value |
|---|---|
| `playbook` | `swebench_verified` (`harbor://swebench-verified@1.0`) |
| agent | `openhands-sdk` (playbook; `agent_kwargs: {max_iterations: 200, version: "1.17.0"}`) |
| scope | 500 Python tasks × `repeats: 5` |
| `max_concurrent` / `sandbox.concurrency` | `15` |
| `solver` | `timeout_strategy: max`, `run_timeout: 10800` (3h), `agent_kwargs.llm_kwargs.timeout: 3600` |
| `sandbox.region` | `us-east-2` |
| `sandbox.ecr_repository` | `${HARBOR_SWEBENCH_ECR_REPOSITORY}` (dedicated `harbor-swebench` repo, **us-west-2**, regardless of sandbox region) |
| `cluster.eval_image` | `${NEL_NEXT_EVAL_IMAGE}` (needs **≥ `0.3.1.1-harbor`** — FEP-1085 reasoning fix) |
| `cluster.container_env.AWS_DEFAULT_REGION` | `us-east-2` (match `sandbox.region`) |
| `instruction_template` | **must be MOUNTED** — the harbor image doesn't bundle the built-in (gotcha below) |

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
`FileNotFoundError: instruction_template not found`. Mount it (the canonical
config uses the compeval OpenHands prompt; the public built-in from the host venv
works too):

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
plus `turn_counter`. Full stack on `services.<svc>.proxy.interceptors`:

```yaml
proxy:
  request_timeout: 3600
  extra_body: {skip_special_tokens: false}   # add model-card sampling extras if the card sets them
  interceptors:
    - {name: system_message, config: {strategy: replace, system_message: "<the OpenHands prompt from bench.yaml>"}}
    - {name: turn_counter, config: {max_turns: 200}}
    - {name: consolidate_system}
    - {name: drop_params, config: {params: [max_tokens, max_completion_tokens]}}
    - {name: reasoning}          # reasoning models: normalize reasoning field …
    - {name: reasoning_replay}   # … and replay it across turns (drop both for instruct)
```

## Score Extraction

Report **`pass@1`** only — benchmark `swebench-verified@1.0`, scorer `pass@1` (0–1):
the resolved rate over the 500 tasks, **already averaged over repeats** (nel-next
reports a single `pass@1`; there is **no `avg-of-N` key** like the 0.2.6 nemo-skills
metrics). MLflow logs it as `pass_at_1`. Read from `report.md` (Benchmark / Scorer
table) in the run dir or `nel eval report -r <run_id>`. Keep `timeout_strategy` +
the instruction/system prompt fixed across baseline vs quantized for a valid delta.
