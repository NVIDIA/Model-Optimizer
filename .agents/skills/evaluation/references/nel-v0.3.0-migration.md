# NEL v0.3.0 Migration

## NEL 0.2.x (launcher) vs 0.3.x (engine)

| | **0.2.x launcher** (`nemo_evaluator_launcher`) | **0.3.x engine** (`nemo_evaluator`) |
| --- | --- | --- |
| Benchmark source | each task runs in its **own published eval-factory container** (`nvcr.io/nvidia/eval-factory/*`); you never build an eval image | **one** `cluster.eval_image`; benchmarks resolve from Python `@register()` modules |
| Where harnesses live | baked into per-task containers upstream | **you pip-install** external harnesses (`nemo-skills`, `lm-eval`) **into the one eval image** |
| Benchmark families | eval-factory `container/` adapters, `ns_*` nemo-skills | **built-in** (17, `nel list -s builtin`), **`skills://`** (nemo-skills), **`lm-eval://`**, + legacy **`container://`** (BC) |
| CLI | `nel run` / `ls` / `status` / `info` | `nel eval run` / `list` / `export` |
| Config schema | Hydra `deployment:` / `evaluation:` / `execution:` | `services:` / `benchmarks:` / `cluster:` / `output:` (one file) |
| Serving | a `deployment:` block | integrated under `services.model` |
| "with tools" | task-specific | `solver: {type: tool_calling, sandbox_tools: true}` + a `sandbox:` |

### Comparing the Config File Structure

**0.2.x launcher** — Hydra, several top-level blocks (+ a `defaults:` preset chain), typically one file per task:

```yaml
defaults: [...]              # Hydra preset chain (resolved at launch)
execution:                   # SLURM: account, partition, walltime, num_nodes, output_dir, sbatch flags
deployment:                  # serving: image, hf_model_handle, served_model_name, TP/PP/DP, extra_args
target:                      # how to reach the served model
  api_endpoint: {...}
evaluation:                  # the benchmark(s)
  tasks: [...]               #   task names live here
export:                      # mlflow export
  mlflow: {...}
```

**0.3.x engine** — one self-contained file, four blocks, all tasks in a single `benchmarks:` list:

```yaml
services:                    # serving (replaces `deployment:` + `target:`)
  model: {type: vllm, model: ..., tensor_parallel_size: 4, extra_args: [...]}
benchmarks:                  # the tasks (replaces `evaluation.tasks:`), one `- name:` each
  - name: gpqa               #   prefix picks the backend (see matrix below)
cluster:                     # SLURM (replaces `execution:`): type, account, walltime, shards, eval_image
output:                      # results + export (replaces `export:`)
  dir: ...
  export: [mlflow]
  export_config: {mlflow: {...}}
```

Key moves:
- `deployment:` + `target:` → **`services.model`** (serving + endpoint unified; the engine wires it to a benchmark via `solver.service`)
- `evaluation.tasks:` → **`benchmarks:`** (a list of `- name:`; the prefix selects the backend)
- `execution:` → **`cluster:`** (adds `shards:` for multi-node sharding and `eval_image:` for the one eval container)
- `export:` → **`output.export` / `output.export_config`**
- `defaults:` (Hydra presets) → **gone** — one explicit file, no preset resolution step


### Backward supporting v0.2.x

0.3.x keeps a backward-compat path for v0.2.x's eval-factory `container/` task variants (e.g.
`mmlu_pro_aa_v3`, `gpqa_diamond_aa_v3`) — though they have **no _native_ 0.3.x benchmark of the same
name**. Two options:

1. run them via the **legacy `container://` backend** (`solver: {type: container}`, which injects a v1-format `run_config.yaml` so v1 configs port with minimal changes — but **`cluster.shards` is unsupported** for legacy container runs)

2. **recommended** — switch to the native built-in (`mmlu_pro`, `gpqa`) or `skills://` equivalents (sharding + one eval image). Note the launcher's AIME entry was already `ns_aime2025` (nemo-skills, *not* a container variant), so it ports directly to `skills://aime25`.


## NEL v0.3.0 Backend Support Matrix

### Supported Backends

NEL 0.3.x resolves a benchmark through one of several **backends**, chosen by the `name:` prefix you
put in `benchmarks:`. The same benchmark is often reachable from **more than one** backend, and the
prefix picks which harness actually runs — which also affects scoring fidelity (whether a benchmark is
scored correctly vs silently mis-scored). That's detailed per-benchmark in **Supported Benchmarks** below.

```text
backend     name: in the config             find available names
----------  ------------------------------  -------------------------------------------------------
built-in    <bench>            (e.g. gpqa)   `nel list -s builtin`
skills://   skills://<dataset>               `nel list -s skills` (after prep) / nemo_skills/dataset/
lm-eval://  lm-eval://<task>                 `lm_eval --tasks list`
gym://      gym://<host:port>?protocol=      Gym repo benchmarks/ + resources_servers/
              native&data=<tasks.jsonl>      (needs a running server → github.com/NVIDIA-NeMo/Gym)
```

Example — a `benchmarks:` block, one entry per benchmark. The `name:` prefix selects the backend (it is **not** uniformly `{backend}://{benchmark}`): built-in has **no prefix** (just `{benchmark}`); `skills://` and `lm-eval://` are `{backend}://{benchmark}`; `gym://` points at a running server, `gym://{host:port}?...&data=<tasks.jsonl>` (the benchmark lives in the data file, not after the prefix):

```yaml
benchmarks: # the benchmarks block
  - name: gpqa                                                   # built-in
  - name: skills://aime25                                        # NeMo-Skills
  - name: lm-eval://hellaswag                                    # lm-eval
  - name: gym://127.0.0.1:8000?protocol=native&data=tasks.jsonl  # NeMo-Gym (running server)
```

### Supported Benchmarks

The list of supported benchmarks are from three NVIDIA-NeMo repositories:
- `NVIDIA-NeMo/Evaluator`: built-in engine [`@register`](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/src/nemo_evaluator/benchmarks)
- `NVIDIA-NeMo/Skills`: **`skills://`** from [`nemo_skills/dataset/`](https://github.com/NVIDIA-NeMo/Skills/tree/main/nemo_skills/dataset),
- `NVIDIA-NeMo/Gym`: **`gym://`** from Gym [`benchmarks/`](https://github.com/NVIDIA-NeMo/Gym/tree/main/benchmarks)+[`resources_servers/`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers).

The matrix below covers only these three backends. **`lm-eval://`** (e.g. `lm-eval://hellaswag`) is also
supported but not enumerated here — discover its tasks with `lm_eval --tasks list`. Only the **built-in**
column is verified against this engine (`nel list -s builtin`); the **`skills://`** and **`gym://`**
columns are compiled from the upstream repos and have **not** all been run — treat them as a routing
guide, not a guarantee (see the per-row notes).


> **`†`** = `skills://` **may mis-score** this **judge / batch / code** benchmark: its bridge ([`skills.py`](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/src/nemo_evaluator/environments/skills.py)) only scores math/multichoice; other types fall back to exact-match → ~0. Use `gym://` (or the nemo-skills container / 0.2.x launcher) instead.

```text
benchmark                  built-in  skills://    gym://   notes
-------------------------  --------  ----------   ------   ----------------------------------------
KNOWLEDGE / QA
  gpqa                       yes       yes         yes      gym: gpqa (+ gpqa_diamond server)
  mmlu                       yes       yes         yes
  mmlu_pro                   yes       yes         yes
  simpleqa                   yes       yes         yes
  triviaqa                   yes       —           —        built-in only
  drop                       yes       —           —        built-in only
  healthbench                yes       —           —        built-in only
  hotpotqa                   —         yes         yes      gym: hotpotqa_qa / hotpotqa_closedbook
  supergpqa                  —         yes         yes
  hle                        —         yes†(judge)  yes      gym recommended (skills may mis-score)
  omniscience                —         yes†(judge)  yes      gym recommended (skills may mis-score)
  critpt                     —         yes†(judge)  yes      gym recommended (skills may mis-score)
MATH
  math500                    yes       yes         yes      skills/gym: math-500
  gsm8k                      yes       yes         yes
  aime25                     —         yes         yes
  aime24                     —         yes         yes
  hmmt_feb25                 —         yes         yes
  polymath                   —         yes         yes
  putnam_bench               —         yes         yes
  minif2f                    —         yes         yes
  proofnet                   —         yes         yes
  ugphysics                  —         yes         yes
  physics                    —         yes         yes
  minerva_math               —         yes         —        skills only
  olympiadbench              —         yes         —        skills only
  omni_math                  —         yes         —        skills only
CODE / SWE
  humaneval                  yes       yes†(code)   yes      built-in faithful; skills code-exec=batch
  mbpp                       —         yes†(code)   yes
  livecodebench              —         yes†(code)   yes      gym recommended (skills may mis-score)
  bigcodebench               —         yes†(code)   yes      gym recommended (skills may mis-score)
  scicode                    —         yes†(code)   yes      gym recommended (skills may mis-score)
  ioi                        —         yes†(code)   yes
  cvdp                       —         yes†(code)   yes
  swe-bench                  —         yes         —        gym has only SWE-RL servers (swe_pivot/swerl_*)
  bird_sql                   —         —           yes      gym only
  spider2_lite               —         —           yes      gym only
  code_gen                   —         —           yes      gym only
  evalplus                   —         —           yes      gym only
INSTRUCTION / FORMAT
  ifbench                    —         yes†(batch)  yes      gym recommended (skills may mis-score)
  ifeval                     —         yes†(batch)  yes      gym recommended (skills may mis-score)
  instruction_following      —         —           yes      gym only
  structured_outputs         —         —           yes      gym only
  multichallenge             —         —           yes      gym only
LONG-CONTEXT
  ruler                      —         yes         yes
  mrcr                       —         yes         yes
  longbench_v2               —         yes         yes
  longcodebench              —         yes         yes
  aalcr                      —         yes†(judge)  yes      gym recommended (skills may mis-score)
AGENTIC / TOOL / RL  (mostly gym-only; the few built-ins are noted)
  terminal-bench-v1          yes       —           —        built-in; agentic terminal tasks
  terminal-bench-hard        yes       —           —        built-in (+ -aa-split variant)
  nmp_harbor                 yes       —           —        built-in (harbor packaging)
  tau2                       —         —           yes
  gdpval                     —         —           yes
  browsecomp                 —         —           yes
  mcqa                       —         —           yes
  blackjack                  —         —           yes
  gymnasium                  —         —           yes
  reasoning_gym              —         —           yes
  arc_agi                    —         —           yes
  aviary                     —         —           yes
  google_search              —         —           yes
  tavily_search              —         —           yes
  xlam_fc                    —         —           yes
  ns_tools                   —         —           yes
  swe_pivot                  —         —           yes
  pinchbench                 yes       —           —        built-in only (NOT gym)
ARENA / JUDGE
  arena_hard                 —         yes†(judge)  yes      the "LM Arena" family
  arena_hard_v2              —         yes†(judge)  yes
  arena_judge                —         —           yes      gym only
  genrm_compare              —         —           yes      gym only
SAFETY
  xstest                     yes       —           yes
  abstention                 —         —           yes      gym only
  over_refusal_detection     —         —           yes      gym only
  jailbreak_detection        —         —           yes      gym only
  indirect_prompt_injection  —         —           yes      gym only
MULTILINGUAL / MULTIMODAL / AUDIO
  mgsm                       yes       yes         —        built-in (multilingual GSM8K)
  mmmlu                      —         yes         yes
  flores200                  —         yes         yes
  wmt24pp                    —         yes         yes
  mmmu_pro                   —         yes         —        skills only (multimodal)
  covost2                    —         yes         —        skills only (audio)
  fleurs                     —         yes         —        skills only (audio)
  labbench2_vlm              —         —           yes      gym only (multimodal)
  vlm_eval_kit               —         —           yes      gym only (multimodal)
```


## Launch NEL v0.3.0

The launch pattern follows the **backend**, not the category. Map each matrix category to the matching
section below:

```text
section (the ### headings below)   categories / when
---------------------------------  --------------------------------------------------------------
built-in and skills:// (native)    KNOWLEDGE/QA · MATH · LONG-CONTEXT · MULTILINGUAL · SAFETY
                                   (every row `yes` under built-in or skills with NO †)
gym:// (server + reward)           AGENTIC/TOOL/RL (gym-only) + every † row (CODE/SWE,
                                   INSTRUCTION/FORMAT, ARENA/JUDGE, LONG-CONTEXT aalcr)
legacy container://                v0.2.x `*_aa_v3` eval-factory container tasks
```

> **Multimodal / audio — not validated here.** Our migration only covered text benchmarks. The engine
> registers a `vlmevalkit://` backend (`environments/registry.py` → `VLMEvalKitEnvironment`), and
> multimodal/audio datasets (e.g. `mmmu_pro`, `covost2`, `fleurs`) may be reachable via `skills://` or
> `gym://` — but we ran **none** of them, so the exact name→backend routing is unconfirmed. Treat as a
> starting pointer, not a tested recipe.

### Basic Usage

Core 0.3.x CLI — the same for any benchmark; the backend (the three subsections below) only changes what goes in `benchmarks:`.

```bash
# Check the version
nel --version                             # v0.3.x
# Config mode (recommended): one file with services + benchmarks + cluster + output
nel eval run config.yaml
nel eval run config.yaml --dry-run        # generate scripts, don't run (inspect per-shard configs)
nel eval run config.yaml --resume         # resume a partial / timed-out run

# Quick mode: a single built-in benchmark against an already-served model
nel eval run --bench gpqa --model-url http://localhost:8000/v1 --model-id my-model --api-key dummy

# Discover names
nel list -s builtin            # also: -s skills  /  -s lm-eval

# Validate before a long run: serve + run a few samples (catches serving/config errors cheap)
nel validate config.yaml

# Track / control a submitted run
nel eval jobs                  # list tracked runs
nel eval status                # progress of the current/last run
nel eval logs                  # tail logs
nel eval stop                  # cancel

# Sharded run: after all shards finish, merge then export
nel eval merge <output.dir>/<run-id>
nel export <bundle> [<bundle> ...] --dest mlflow -o tracking_uri=<uri> -o experiment_name=<exp>  # takes 1+ bundles
```

### Single-node vs sharded

Sharding is **orthogonal to the benchmark config** — the `services:` / `benchmarks:` / `solver:` blocks
shown in the backend subsections below are identical either way. The only difference is one line in the
`cluster:` block plus a post-run merge:

```text
                  single-node (no shards)            sharded (cluster.shards: N)
----------------  ---------------------------------  ----------------------------------------
config diff       cluster.shards absent (or = 1)     cluster.shards: N      (one added line)
benchmarks/solver identical                          identical
how it runs       1 worker serves the model,         N independent single-node TP workers, each
                  runs ALL (problems × repeats)      serves its OWN model copy, runs 1/N of them
launch            nel eval run config.yaml           nel eval run config.yaml   (same command)
results           one result bundle per benchmark    N shard results -> nel eval merge <dir>/<run-id>
export            nel export <bundle> ...            nel export <merged-bundle> ...
when              small / quick runs                 the long pole (e.g. aime25 repeats=64); ~N× faster
```

Key points:

- **`repeats` is *not* divided — the *problem set* is.** Each shard runs a disjoint slice of all
  `problems × repeats` tasks, so the merged result is mathematically identical to a single-node run,
  just ~N× faster wall-clock.
- **This is the throughput mechanism, not data-parallel serving.** Sharding launches N separate
  single-node jobs (each its own model instance); it does **not** rely on vLLM data-parallel (DP is
  fragile on some stock vLLM builds — DP can fail in engine-core init).
- **The only extra step is `nel eval merge <dir>/<run-id>`** — it stitches the N shard outputs into one
  bundle per benchmark before you `nel export` (see "Export results to MLFlow" below).
- **Works cleanly for native (built-in / `skills://`).** Legacy `container://` is single-node only. A
  `gym://` run *can* shard but it isn't a free swap — its one shared resource server must be at a
  routable host (not loopback) reachable from every shard (see the `gym://` section); the validated gym
  path is single-node `cluster: {type: local}`.

### The `solver:` block — `simple` vs `tool_calling`

`solver:` sets **how the model answers each problem**; `service:` names which `services:` entry to run
against (e.g. `service: model`).

- **`simple`** — one model call per problem, no tools. For pure reasoning (knowledge / math / multichoice).
- **`tool_calling`** — a multi-turn ReAct loop: the model calls tools that a `sandbox:` block runs, up to
  `max_turns` (turn-exhaustion = a failed episode). Needs `sandbox_tools: true` + a `sandbox:`, and a model
  served with tool calling enabled (vLLM: `--enable-auto-tool-choice` + `--tool-call-parser`). For
  tool-augmented tasks (run code, compute, search).

**Backend and solver are independent.** A native built-in / `skills://` benchmark can run *either* solver
— e.g. `gpqa` with `simple` (no-tools) or with `tool_calling` + a `local` sandbox (the "with-tools"
variant). `tool_calling` is **not** exclusive to `gym://`.


### built-in and skills:// (native)

The common path — every matrix row that is `yes` under **built-in** or **skills** with **no `†`**. The
engine serves the model itself and scores per-sample; just list the names in `benchmarks:` with a
`simple` solver (no sandbox, no server). Built-in takes a bare name; nemo-skills takes the `skills://`
prefix.

Built-in benchmarks need nothing extra. For **`skills://`**, first install **NeMo-Skills** into your eval
image — `pip install nemo-skills` from [NVIDIA-NeMo/Skills](https://github.com/NVIDIA-NeMo/Skills); see the
repo README. It's baked into `cluster.eval_image` (and the dataset is prepped there). Likewise `lm-eval://`
needs `lm-eval` in the image.

```yaml
benchmarks:
  - name: gpqa                       # built-in (multichoice)
    repeats: 8
    max_concurrent: 64
    solver: {type: simple, service: model}
  - name: mmlu_pro                   # built-in (multichoice)
    repeats: 1
    max_concurrent: 64
    solver: {type: simple, service: model}
  - name: skills://aime25            # nemo-skills (math)
    repeats: 64
    max_concurrent: 256
    solver: {type: simple, service: model}
  # WITH-TOOLS variant of the SAME native benchmark — backend unchanged, just swap the solver and add a
  # sandbox (see "The solver: block" above). Use instead of the simple gpqa entry to give it tools:
  # - name: gpqa
  #   repeats: 8
  #   max_concurrent: 64
  #   solver: {type: tool_calling, service: model, sandbox_tools: true, max_turns: 100}
  #   sandbox: {type: local, concurrency: 64}

cluster:
  shards: 8                          # N single-node workers, each runs 1/N of (problems × repeats); `nel eval merge` after
  # ... + type/account/walltime/eval_image/node_pools (see r030_example_eval.yaml)
```

`nel eval run config.yaml`. `repeats` matches your model card's sampling (e.g. gpqa avg-of-8, aime25 avg-of-64);
sampling (`temperature`, `top_p`, `max_tokens`) lives under `services.model.generation`. Full runnable
config: [`recipes/examples/r030_example_eval.yaml`](../recipes/examples/r030_example_eval.yaml)
(simple / no-tools). For a tool-calling benchmark see the `gym://` example below.

The native path runs under `cluster: {type: slurm}` with `auto_resume: true` (the default), so NEL
**auto-resumes** a wall-clock kill itself — it chains a successor SLURM job, no manual step. (A `gym://` run
is different: it runs under `cluster: {type: local}`, which has no `auto_resume`, so you resume it by hand —
see the gym section's "Resume the jobs".)

### gym:// (server + reward)

`gym://` runs a NeMo-Gym **resource server** that scores each response — the faithful path for the `†`
benchmarks `skills://` mis-scores and for agentic envs. We run it as a **two-job split**: the model on a
GPU node (its own serve job), and the grader + evaluator on a CPU node. Why split: serving needs the vLLM
container, while `nel` and the Gym server live in their own venvs (see the last caveat in §7).

#### 1. The pieces (two jobs)

```text
program                        what it does                                          runs on
-----------------------------  ----------------------------------------------------  ------------------------------
the grader (gym env start)     checks each answer, returns a score                   CPU node (you start it there)
the conductor (nel eval run)   sends questions -> model, answers -> grader, tallies   CPU node — in-process
                                                                                      (cluster: {type: local})
the model                      answers the questions                                 GPU node — its own vllm serve
                                                                                      job (services.model.type: api)
```

Both the grader and the conductor run on the CPU node that `gym_eval.sbatch` allocates (`--partition=cpu`,
§6) — `cluster: {type: local}` means NEL runs the conductor **in-process there, not on the login node**. The
conductor reaches the **model** over the network (the GPU node's hostname) and the **grader** at `127.0.0.1`
(same CPU node).

#### 2. Start the gym grader + get its port

First install NeMo-Gym (the `gym`/`ng` CLI) — clone [NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym)
and `uv venv --python 3.12 && uv sync`; see the repo README.

The grader is **not** started by NEL — you launch it on the **CPU node** (same node as the evaluator, so
it's reachable at `127.0.0.1`). A benchmark's bundled config (`resources_servers/<name>/configs/<name>.yaml`)
usually packs **more than one server instance** into one file — you only start one of them. For
[ifbench](https://github.com/NVIDIA-NeMo/Gym/blob/main/resources_servers/ifbench/configs/ifbench.yaml):

```text
top-level key in <name>.yaml    kind                  role                             NEL native: start it?
------------------------------  --------------------  -------------------------------  ----------------------
ifbench                         resources_servers     the GRADER — scores via /verify  KEEP  (all NEL needs)
ifbench_simple_agent            responses_api_agents  a sample gym-driven agent loop   DROP
  └ model_server: policy_model  responses_api_models  the model that agent would run   (placeholder name only;
                                (referenced, NOT          — gym-driven path, not NEL    not defined in the file)
                                 defined in the file)
```

**Keep / drop — which and when:**
- **NEL `gym://…?protocol=native` (this guide)** — NEL *is* the conductor: it calls the model itself, then
  posts each answer to the grader's `/verify`. Start **only the `resources_servers` grader** (`ifbench`);
  drop the agent.
- **gym-driven (Responses API)** — gym runs the `*_simple_agent` itself and needs a real
  `responses_api_models` wired in (the bundle ships `policy_model` as an unresolved placeholder). Different
  path; not what NEL native uses.

`gym env start --resources-server <name>` (and `--config <name>.yaml`) load the **whole** bundled file, so
they also try to start `*_simple_agent` and **fail** on the undefined `policy_model`. To bring up only the
grader, point `--config` at a config that keeps **just** the `resources_servers:` block (drop
`responses_api_agents:`). It registers with a head process on `:11000` and gets a **dynamic** port:

```bash
export RAY_TMPDIR=/tmp                                          # Lustre socket-path fix
gym env start --config resources_servers/<name>/configs/<name>.yaml &    # Launch gym grader in background by using "&"
curl -s http://127.0.0.1:11000/server_instances | python3 -m json.tool   # -> [{"name": .., "url": ".../<PORT>"}]
```

The port is **assigned dynamically** — read it from `/server_instances` (the `url` field ends in
`:<PORT>`) and drop it into the conductor config's `gym://127.0.0.1:<port>` URI (§4). For scripting, parse it:

```bash
PORT=$(curl -s http://127.0.0.1:11000/server_instances \
       | python3 -c "import json,sys; d=json.load(sys.stdin); print(d[0]['url'].rsplit(':',1)[1] if d else '')")
```

Running more than one server lists them all — match the row by `name`.


#### 3. Get the data

The questions live in a **local `tasks.jsonl`** that the conductor reads (`data=` points at it). Two ways:

**Way 1 - no download (quick):** use the bundled [`example.jsonl`](https://github.com/NVIDIA-NeMo/Gym/blob/main/resources_servers/ifbench/data/example.jsonl) already in the Gym repo (each server has one under
`resources_servers/<name>/data/`; small, prompt field pre-built):

```text
data=<gym>/resources_servers/<name>/data/example.jsonl
```

**Way 2 - full set (downloads from HuggingFace):** run the benchmark's `prepare.py` — one per benchmark dir under [`Gym/benchmarks/`](https://github.com/NVIDIA-NeMo/Gym/tree/main/benchmarks) (e.g. [`benchmarks/ifbench/prepare.py`](https://github.com/NVIDIA-NeMo/Gym/blob/main/benchmarks/ifbench/prepare.py)):

```bash
cd <gym>                                              # your NeMo-Gym clone
export HF_HOME=<cache>/huggingface                    # keep the HF download off your home dir
PYTHONPATH=$PWD .venv/bin/python benchmarks/<name>/prepare.py   # downloads from HF -> benchmarks/<name>/data/
# then point data= at the output, e.g.:
#   data=<gym>/benchmarks/ifbench/data/ifbench_benchmark_eval.jsonl
```

(Native path: each row needs a `responses_create_params` prompt field - bundled `example.jsonl` already
has it; `prepare.py` output may need it added.)


#### 4. The conductor config (what `nel eval run` reads)

This is the config the **conductor** (`nel eval run` — the third piece in §1) reads. It tells the conductor
which **model** to call (§5), which **grader** + **data** to score against (§2/§3), where it runs, and where
to write results. It is *not* the grader's config (that's §2's gym server) — it's the run itself.

Its placeholders are filled by the other steps — the grader port (§2), the data path (§3), and the model host (§5):
model = external endpoint (`type: api`), benchmark = the local gym grader, evaluator in-process
(`cluster: {type: local}`):

```yaml
services:
  model:
    type: api
    url: http://<gpu-host>:8000/v1/chat/completions   # <gpu-host> from §5 (serve_host.txt)
    model: <id>
    api_key: dummy
    generation:
      temperature: 1.0
      top_p: 0.95
      max_tokens: 32768            # reasoning model: big enough to finish reasoning + emit the answer
benchmarks:
  # simple = single-shot (ifbench / ifeval); for agentic envs use `tool_calling` + a `sandbox:` block.
  - name: "gym://127.0.0.1:<port>?protocol=native&data=<gym>/.../tasks.jsonl"   # <port> from §2, data from §3
    repeats: 1
    max_concurrent: 64
    solver: {type: simple, service: model}
cluster:
  # gym's single grader lives on 127.0.0.1, so a gym run can't shard across nodes — keep it single-node.
  type: local                      # evaluator runs in-process on the CPU node (no shards)
output:
  dir: <rundir>
```

Full runnable config: [`recipes/examples/r030_gym.yaml`](../recipes/examples/r030_gym.yaml).


#### 5. Serve the model on a GPU node

`type: api` means you serve the model yourself — a plain `vllm serve` in its own GPU `sbatch` job (the
vLLM **container**). Publish the node's hostname so the eval job can build `url`:

```bash
#!/bin/bash
# serve.sbatch
#SBATCH --job-name=serve-policy
#SBATCH --partition=<gpu>
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --exclusive
hostname > <shared>/serve_host.txt                # publish the node for the eval job
srun --container-image <vllm.sqsh> \
  --container-mounts <model-dir>:/model:ro,<cache>:/cache \
  bash -lc 'vllm serve /model --served-model-name <id> --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 4 --max-model-len 262144 --trust-remote-code \
    --reasoning-parser <p> --tool-call-parser <p> ...'
```

- **`--host 0.0.0.0`** (not `127.0.0.1`) so the CPU eval node can reach it cross-node; the GPU node serves only the model.
- **`<gpu-host>` is resolved at runtime** — you don't know the node until SLURM dispatches the job, so the serve job writes its `hostname` to a shared file and the eval job reads it (`POLICY_HOST=$(cat <shared>/serve_host.txt)`) to fill `url`.


#### 6. Run it — the two-job pattern

From the login node you submit **two** jobs — `serve.sbatch` (§5) and `gym_eval.sbatch` (below):

```bash
sbatch serve.sbatch       # GPU: §5 — vllm serve (publishes serve_host.txt)
sbatch gym_eval.sbatch    # CPU: the script below
```

`gym_eval.sbatch` bundles §2 (start grader + get port) + the host hand-off + the conductor (§4 config):

```bash
#!/bin/bash
# gym_eval.sbatch
#SBATCH --job-name=gym-eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
export RAY_TMPDIR=/tmp                                         # Lustre socket-path fix
# 1. start the grader (§2) on this node, then wait for it to register + read its DYNAMIC port
cd <gym>
# Launch gym grader in background by using "&". Keep only resources_servers and drop the agent (§2) as needed
gym env start --config resources_servers/<name>/configs/<name>.yaml &
PORT=""
for i in $(seq 1 60); do                                      # server takes a few s to register
  PORT=$(curl -s http://127.0.0.1:11000/server_instances \
         | python3 -c "import json,sys; d=json.load(sys.stdin); print(d[0]['url'].rsplit(':',1)[1] if d else '')")
  [ -n "$PORT" ] && break; sleep 5
done
[ -z "$PORT" ] && { echo "gym server did not come up"; exit 1; }
# 2. wait for the GPU serve (serve_host.txt + model live), then fill the §4 config placeholders
until [ -f <shared>/serve_host.txt ]; do sleep 5; done
POLICY_HOST=$(cat <shared>/serve_host.txt)
until curl -s --max-time 8 "http://$POLICY_HOST:8000/v1/models" | grep -q '"id"'; do sleep 10; done
sed -i "s#<gpu-host>#$POLICY_HOST#; s#<port>#$PORT#" config.yaml
# 3. run the conductor (§4); ${RESUME} is empty on a fresh run, "--resume" when resuming (§7)
nel eval run config.yaml ${RESUME:-}
```

Submit `serve.sbatch` first (a plain `sbatch --dependency=after` only waits for the serve job to *start*,
not for vLLM to finish loading — that's why `gym_eval.sbatch` polls `…/v1/models` above before running).

#### 7. Resume the jobs

Resuming a wall-clock kill is **manual** here. `cluster: {type: local}` has no `auto_resume` (that only
chains `slurm`-cluster jobs — see the native section), so neither job restarts itself, and the GPU serve
job dies too. A resume is therefore **two steps** — relaunch the serve, then resume the eval:

```bash
# 1. relaunch the GPU serve — it rewrites serve_host.txt with the new node's hostname
sbatch serve.sbatch

# 2. resume the eval — it waits for the fresh serve_host.txt + /v1/models, then continues
sbatch --export=ALL,RESUME=--resume gym_eval.sbatch          # RESUME -> nel eval run … --resume (§6 step 3)
```

Completed rollouts are checkpointed per `(problem, repeat)`, so the resume **skips everything already
scored** and re-runs only what was in flight — even though the new serve node changes the model URL. (NEL
keys the skip on its *verified* log, which survives the config-hash change; only the un-scored inference
cache is dropped and regenerated.)

#### 8. Caveats

- **Reasoning models need a big enough `generation.max_tokens`** — too small truncates mid-reasoning, the
  grader sees empty output, and the score collapses.
- **Reasoning models: enable thinking via `proxy.extra_body`** — pass `chat_template_kwargs: {enable_thinking:
  true}` under `services.model.proxy.extra_body` (it's merged into every request) so the chat template emits
  the reasoning block. This is a *model* setting (applies to native runs too), not gym-specific.
- **`example.jsonl` is a small subset** — use `prepare.py` (§3) for the full benchmark.
- **Sharding:** a gym run has one shared grader, so it doesn't shard across nodes cleanly (loopback breaks)
  — keep it single-node.
- **Why not "one GPU node, NEL serves the model" (`type: vllm` + `cluster: {type: local}`)?** NEL's local
  path runs `python -m vllm` from the `nel` venv (it ignores the container `image:`), and that venv has no
  vLLM — so it errors at model startup. Serving needs the vLLM container, which is why §5 is a separate job.


### legacy container://

Backward-compat path for v0.2.x's eval-factory `*_aa_v3` container tasks that have no native 0.3.x
benchmark. The `container` solver injects a v1-format `run_config.yaml`, so v1 configs port with
minimal change — but **`cluster.shards:` is unsupported** for legacy container runs (single-node only).

```yaml
benchmarks:
  - name: mmlu_pro_aa_v3             # v0.2.x eval-factory container task
    solver: {type: container, service: model}

cluster:
  # shards: N                        # UNSUPPORTED for legacy container — single-node only (omit it)
  # ... + type/account/walltime/eval_image/node_pools
```

Prefer the native built-in / `skills://` equivalent (`mmlu_pro`, `gpqa`) when one exists — you get
sharding + the one eval image. Use `container://` only when there is no native port.


## Export results to MLFlow

SLURM runs do **not** auto-export (the `output.export` block only fires for local runs). After all
shards finish, merge then export from the login node:

```bash
# RUN = the run dir: <output.dir>/<run-id> (one timestamped subdir per run; holds shard_0/ ... shard_N/)
RUN=<output.dir>/<run-id>

# 1. merge the N shards -> writes the merged per-benchmark bundles BACK INTO $RUN (alongside the shards)
nel eval merge "$RUN"

# 2. export each merged benchmark bundle ($RUN/<bench>) to MLflow
nel export "$RUN/gpqa" "$RUN/mmlu_pro" "$RUN/skills___aime25" --dest mlflow \
  -o tracking_uri=<your-mlflow-uri> \
  -o experiment_name=<exp>
```

Both commands use the **same** run dir (`<output.dir>/<run-id>`): merge reads its `shard_*/`
subdirectories and writes the merged `<bench>/` bundles next to them, which export then consumes. Tags
from `output.export_config.mlflow.tags` (e.g. `model`, `checkpoint_path`, `num_nodes`, `precision`,
`variant`) ride along and feed your MLflow dashboards / downstream viewers. The `/` in a
`skills://` name becomes `___` in the on-disk bundle dir (`skills___aime25`).

> **Always set a stable `model` (and ideally `checkpoint_path`) tag.** Dashboards key/group runs by
> these tags to attribute a score to a model or checkpoint. A run exported **without** them can't be
> attributed — it shows up as an orphan/"no checkpoint" row instead of landing on the right model — and
> a `model` value that drifts between runs (or differs from a baseline's) splits what should be one row.
> The engine logs only a generic headline metric (`pass_at_1` / `mean_reward`); the benchmark identity
> rides in the `benchmark` tag and the model identity in `model`, so both tags are what make the export
> usable downstream.

Example `output` block with the tags set (lives in the run config; `nel export` also accepts the same
tracking URI / experiment via `-o`):

```yaml
output:
  dir: <run dir>
  export: [mlflow]
  export_config:
    mlflow:
      tracking_uri: <your-mlflow-uri>
      experiment_name: <exp>
      tags:
        model: <model-name>          # set this — dashboards group/attribute runs by it
        checkpoint_path: <abs path>  # the per-checkpoint row label downstream
        benchmark: <bench>           # benchmark identity (the engine logs only a generic metric key)
        precision: bf16
        variant: <run label>         # e.g. base / with-tools / 96k-thinking
```
