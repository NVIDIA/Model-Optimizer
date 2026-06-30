# nel-next (nemo-evaluator 0.3.x) — shared reference for harbor / agentic benchmarks

Agentic AA benchmarks (Terminal-Bench 2.x, SWE-bench Verified/…) — an agent drives
a sandboxed machine, each task graded by its own test script — run on **nel-next**
= `nemo-evaluator` **0.3.x** + `harbor` extra, **not** the default
`nemo-evaluator-launcher` 0.2.6. This file is the common machinery; per-benchmark
deltas live in `recipes/tasks/aa_next/{terminal_bench_2_1,swebench_verified}.md`.
Start configs from `recipes/examples/example_eval_next.yaml`.

## Different system from 0.2.6 — don't mix them

| | default (SKILL Steps 1–9) | nel-next |
|---|---|---|
| package | `nemo-evaluator-launcher` 0.2.6 | `nemo-evaluator[harbor]` 0.3.x |
| env | the skill's normal env | **separate venv** (`.agents/scripts/nel-next.sh`) |
| CLI | `nel run --config X.yaml` | `nel eval run X.yaml [--submit]` |
| overrides | `-o ++a.b.c=v` | `-O a.b.c=v` |
| canary limiter | `++…limit_samples=N` | `-O benchmarks.0.max_problems=N` (NOT `--max-problems`, which is `--bench`-only) |
| schema | execution/deployment/evaluation/export | `services`/`benchmarks`/`cluster`/`output` |

## Separate venv (mandatory)

Installing 0.3.x into the 0.2.6 env clobbers `nel`, so it lives in its own venv:

```bash
.agents/scripts/nel-next.sh --setup-only      # one-time, ~1-2 min (needs `uv`)
.agents/scripts/nel-next.sh eval run <cfg> --dry-run | --submit | …
```

Default install is public PyPI `nemo-evaluator[harbor]==0.3.*`; set
`NEL_NEXT_ORIGIN`/`NEL_NEXT_REF` for the internal git build (see script header).

## Credentials + internal infra (`.env`)

- **AWS creds** (sandboxes run in AWS ECS Fargate, shared NVIDIA acct `463701203462`)
  and `HF_TOKEN` are secrets you add yourself: `HF_TOKEN`, `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`. Never open `.env` with file tools — shell only.
- **Internal harbor infra is NOT hardcoded** — configs read `${NEL_NEXT_EVAL_IMAGE}`,
  `${HARBOR_ECR_REPOSITORY}` (Terminal-Bench), `${HARBOR_SWEBENCH_ECR_REPOSITORY}`
  (SWE-bench) from `.env`. Run **`modelopttools:eval-config`** (Step 3b) to write
  them — it holds the canonical values, arch/region rules, and points to the
  per-benchmark `bench.yaml` source of truth.
- `set -a && source .env && set +a` before running so `${VAR}` resolves.

## Architecture — where each piece runs

| Component | Where |
|---|---|
| submit the sbatch | cluster **login** node (submission only) |
| vLLM server **+** harbor `eval_image` (agent orchestrator) | the **same GPU compute node**, co-located via `srun --overlap` (shared netns → agent reaches vLLM at `localhost:5000`) |
| task sandboxes + grading | **remote AWS ECS Fargate** (acct `463701203462`), reached over SSH tunnels |

`shards: N` → N compute nodes, each running its own vLLM + eval_image pair on 1/N
of the trials; the sbatch auto-merges when all shards finish.

## Config schema (self-contained)

`playbook: <name>` pulls the benchmark's defaults; sibling keys override. All
schemas are `extra="forbid"` (unknown keys hard-fail).

```yaml
services:
  <svc>:
    type: vllm                  # NEL deploys + serves; auto-wrapped as an api service for the agent
    model: /path/to/checkpoint  # bind-mounted to /model:ro
    served_model_name: <name>
    protocol: chat_completions
    port: 5000
    tensor_parallel_size: 8     # STRUCTURED — don't repeat parallelism in extra_args
    data_parallel_size: 1
    num_nodes: 1
    image: vllm/vllm-openai:<ver>   # the vLLM SERVING image (≠ eval_image); bump to recipes.vllm.ai min
    startup_timeout: 3600.0
    extra_args: [...]           # raw vllm flags — everything EXCEPT parallelism/served-model-name/port (see "vLLM deployment" below)
    extra_env: {...}            # VLLM_* backend env (e.g. NVFP4 MoE flags)
    container_mounts: [<lustre>/.cache/vllm:/cache/vllm, ...]
    generation: {temperature: 1.0, top_p: 0.95}
    proxy: {request_timeout: 1800, extra_body: {...}, interceptors: [...]}
    node_pool: gpu
benchmarks:                     # EXACTLY ONE entry — one benchmark per config (see "One benchmark per config")
  - playbook: <benchmark>       # per recipe
    repeats: <N>
    max_concurrent: <N>         # keep == sandbox.concurrency
    solver: {service: <svc>, timeout_strategy: task|max}
    sandbox: {region: <...>, ecr_repository: ${HARBOR_ECR_REPOSITORY}, concurrency: <N>, log_stream_prefix: <...>}
cluster:
  type: slurm
  hostname: <login fqdn>
  username: <user>
  account: <account>
  walltime: "04:00:00"          # auto_resume chains across windows
  shards: 1
  eval_image: ${NEL_NEXT_EVAL_IMAGE}   # from eval-config
  sbatch_extra_flags: {switches: 1, exclusive: true}
  container_env: {HF_TOKEN: ${HF_TOKEN}, HF_HOME: /cache/huggingface, AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID}, AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY}, AWS_DEFAULT_REGION: <region>, LLM_API_KEY: "no-key-needed"}
  mount_home: false
  auto_resume: true
  max_retries: 3
  node_pools: {gpu: {partition: <part>, nodes: 1, ntasks_per_node: 1, gpus_per_node: 8}}   # match TP*DP
output:
  dir: <lustre>/eval-output/<model>-<benchmark>
  export: [mlflow]              # ALWAYS — see below
  export_config: {mlflow: {...}}
```

## vLLM deployment — map SKILL Step 3 onto nel-next fields

Same `recipes.vllm.ai` (exact variant) + model-card / `config.json` cross-check as
SKILL.md Step 3 (same vLLM). The 0.2.6 `command:` maps to structured `services.<svc>`:

| 0.2.6 `command:` | nel-next `services.<svc>` |
|---|---|
| `vllm serve /checkpoint`, `--served-model-name`, `--port` | `model:` (→`/model`), `served_model_name:`, `port:` (auto-emitted) |
| `--tensor/-data/-pipeline-parallel-size`, multi-node | `tensor_parallel_size`/`data_parallel_size`/`pipeline_parallel_size` + `num_nodes` (**not** in `extra_args`) |
| all other serve flags (`--max-model-len`, `--max-num-seqs`, parsers, `--enable-*`, `--model-loader-extra-config`, …) | `extra_args:` |
| `env_vars` (`VLLM_*`, NVFP4 MoE flags) | `extra_env:` |
| `image:` (NVFP4: MiniMax-M2.7 ≥ v0.20.0; sm_103 → `-cu130`) | `image:` (serving image, ≠ `eval_image`) |

Size TP/DP + backend defaults (`--max-num-seqs = ceil(max_parallelism/DP)`, MoE
`--enable-expert-parallel`, …) per `references/parallelism.md` + Step 3. **Sampling
is a mandatory model-card lookup** (Step 3 / `references/model-card-research.md`,
never generic defaults): `generation.temperature`/`top_p` (+`max_tokens` if the card
caps output) and `proxy.extra_body` for any card extras (`skip_special_tokens`, thinking toggles).

## One benchmark per config

Keep `benchmarks:` to a **single** entry — one benchmark per file. Don't combine
benchmarks (`eval_image`/ECR/region/interceptors/instruction-template/repeats/sharding
differ, and the harbor deploy+sandbox flow is per-config); run each as its own config
with its own `run_id`, copying the shared `services:` block.

## Rules & gotchas

- **`eval_image`** = `${NEL_NEXT_EVAL_IMAGE}`. `0.3.1.1-harbor` is multi-arch and is
  the minimum for **TB 2.1**; older `0.17.x/0.18.x-harbor-<arch>` are arch-suffixed.
  Private gitlab-master image → cluster needs enroot creds (SKILL Step 7.5).
- **Mount sources must pre-exist** — pyxis won't create the host side of a bind
  mount (invisible to `--dry-run`, fails at canary). `ssh <login> 'mkdir -p
  <lustre>/<user>/.cache/{vllm,huggingface}'`.
- **`timeout_strategy`**: `task` = each task's own timeout (leaderboard-comparable);
  `max` = `max(task, playbook)`; `override` = always playbook. Match the benchmark's
  canonical `bench.yaml`.
- **MLflow export — always add** `output.export: [mlflow]` + `export_config.mlflow`
  (`tracking_uri: https://mlflow-nemo-evaluator.nvidia.com/`, **hardcoded**
  `experiment_name: <user>/<model>` since `${USER}=root` in-container; tags aligned
  with `recipes/examples/example_eval.yaml`: `framework`/`model`/`temperature`/
  `top_p`). It runs cluster-side at finalize. **Caveat:** the public-0.3.0
  host's SLURM-submit path drops `output.export` from the sidecar (the internal host
  propagates it), and a local `nel export` is SSO-blocked (MLflow needs an auth
  token from the internal run env) — so a link only appears on the internal host /
  with that token.

## Run (dry-run → canary → full) + lifecycle

```bash
set -a && source .env && set +a; NEL=.agents/scripts/nel-next.sh
$NEL eval run <cfg>.yaml --dry-run                                  # validate/render (no SSH)
$NEL eval run <cfg>.yaml --submit -O benchmarks.0.max_problems=2 -O benchmarks.0.repeats=1 -O benchmarks.0.max_concurrent=2   # canary
$NEL eval run <cfg>.yaml --submit                                   # full
$NEL eval {status|logs -f|report -f markdown|merge} -r <run_id>     # lifecycle
```

`eval run` on a slurm cluster scp's the sbatch + redacted `.secrets.env` and
submits via SSH; a built-in afternotok chain auto-resumes across walltime windows;
sharded runs auto-merge.

## Non-fatal noise

- AWS `RegisterTaskDefinition: ThrottlingException` at high concurrency — SDK
  retries (≤8); lower `max_concurrent` if frequent.
- `solver failed — grading 0.0` agent crashes are rare but real (e.g. terminus-2
  tmux `send-keys`) — distinguish from genuine misses + task timeouts; the final
  report breaks them out.
