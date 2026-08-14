# MRCR (OpenAI Multi-Round Co-reference Resolution, NeMo Gym `simple_agent`)

## Task Details

- Upstream benchmark: <https://github.com/NVIDIA-NeMo/Gym/tree/main/benchmarks/mrcr>
- Resource server: <https://github.com/NVIDIA-NeMo/Gym/blob/main/resources_servers/mrcr/configs/mrcr.yaml>
- Dataset: `openai/mrcr` (HF, gated → `HF_TOKEN` required)

MRCR is a **long-context retrieval** benchmark. Each task is a long synthetic
multi-turn conversation containing N near-identical "needle" responses; the final
user turn asks the model to *reproduce the Nth occurrence verbatim, prefixed with
a random string*. Scoring is deterministic: `SequenceMatcher.ratio()` between the
stripped response and the reference answer, **gated on the response starting with
the required random prefix** (wrong prefix → 0). Results are stratified by needle
count (2 / 4 / 8) — accuracy falls sharply as N rises.

**Not an AA benchmark.** It lives under `recipes/tasks/gym/`, not `aa_gym/`, and
is never part of the AA Index v2 set.

It runs on the **0.2.6 `nel` launcher** as a `nemo_gym` task (NOT nel-next), so
Steps 1–9 apply — with the branch differences below.

## What makes MRCR different from GDPVal (the other gym task)

MRCR is **much simpler to configure** than GDPVal — most of the GDPVal machinery
does not apply:

| | GDPVal | MRCR |
| --- | --- | --- |
| Gym agent | `stirrup_agent` | **`simple_agent`** |
| Apptainer SIF sandbox | required | **none** |
| LLM judge | Gemini 3.1 Pro, 4 trials | **none** — deterministic string grading |
| External secrets | `INFERENCE_API_KEY`, `TAVILY_API_KEY`, judge URL | **`HF_TOKEN` only** |
| Dominant cost | agent turns + judge | **context length** (up to 1M tokens/prompt) |

What it shares with GDPVal: it is **standalone** (one gym eval per config — never
add MRCR to a multi-task `evaluation.tasks` list, and never add other tasks to an
MRCR config), it needs `NEMO_EVALUATOR_TRUST_PRE_CMD=1` (the config has a
`pre_cmd`), and `limit_samples` behaves differently from the normal task path.

## Config

Start from the self-contained example and edit it — **do not** copy a fragment
into another config:

```text
recipes/examples/gym_mrcr/example_gym_mrcr.yaml   # SLURM + vLLM, 1M variant, self-contained
```

### Variant selection — pick before anything else

Three upstream variants; the choice sets the context envelope, the dataset file,
**and the metric key prefix**. The reviewed golden uses the **1M** variant.

| Gym config path | Tokenizer / cap | Agent + metric prefix | `num_repeats` |
| --- | --- | --- | --- |
| `benchmarks/mrcr/config_n3_1m.yaml` | NVIDIA Nemotron (gated), ≤ 1,048,576 tok | `mrcr_n3_1m_benchmark_simple_agent` | 1 |
| `benchmarks/mrcr/config_n3_128k.yaml` | NVIDIA Nemotron (gated), ≤ 131,072 tok | `mrcr_n3_128k_benchmark_simple_agent` | 1 |
| `benchmarks/mrcr/config.yaml` | `o200k_base`, **no cap** | `mrcr_benchmark_simple_agent` | 4 |

The n3 variants *drop* samples whose tokenized conversation exceeds the cap, so
the two n3 datasets are different sizes and **their scores are not comparable to
each other or to the plain variant**. Pick one and keep it fixed across baseline
and candidate.

Change the variant in **both** `data_prep_params` and `collect_rollout_params`
(`+config_paths=[...]`) — changing only one silently prepares one dataset and
rolls out another.

`num_repeats` comes from the chosen variant's upstream config and the template
does not override it. Per the upstream README, for `type: benchmark` datasets the
in-config `num_repeats` is a **placeholder** that only duplicates rows for
`train`/`validation` splits — the real repeat count comes from the runner. The 1M
golden reports `pass@1`, i.e. one rollout per task. **Do not change repeat counts
when aligning a run to a golden.**

### Serving envelope (1M variant)

The long-context envelope, not the model, drives these:

- `--max-model-len 1100000` **plus** `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` in
  `deployment.env_vars` — vLLM refuses a `--max-model-len` above the checkpoint's
  declared `max_position_embeddings` without it.
- `gpu_memory_utilization: 0.95` (up from the usual 0.85) to fit the KV cache.
- `--enable-prefix-caching` + `--enable-chunked-prefill` + a large
  `--max-num-batched-tokens` (golden: 131072).
- `--kv-cache-dtype fp8` is part of the golden envelope. **It is itself a
  precision choice** — keep it identical between baseline and quantized runs or
  the delta also measures KV-cache quantization.
- Fan out with `execution.num_nodes` / `num_instances` (golden: **4 / 4**,
  HAProxy pattern A — see `references/multi-node.md`). `parallelism` is the
  **total** gym client concurrency across all instances, so
  `--max-num-seqs = ceil(parallelism / num_instances / DP)` (golden: 256/4/1 = 64).

### Output length — do not cap it

MRCR answers **reproduce an entire earlier assistant turn verbatim**. Any output
cap truncates the reproduction and craters `SequenceMatcher.ratio()`. The golden
sets `max_new_tokens: null` and passes
`++responses_create_params.max_output_tokens=null`. Leave both uncapped.

### `pre_cmd` — required

The pinned Gym commit's MRCR prepare imports `tiktoken` at module load and falls
back to `transformers.AutoTokenizer` for the N3 tokenizer path; **neither is a
declared dep of that commit's Gym venv**, so prepare dies with `ModuleNotFoundError`
without the `pre_cmd` that installs them. Needs `NEMO_EVALUATOR_TRUST_PRE_CMD=1`
in the launching shell.

### Gym commit ↔ container coupling — read before trusting a score

The template pins Gym to `a431501aa294f3237d472aaf58dd1e5026156ea8`, the commit
the golden validated. **MRCR depends on the pin actually applying**: the N3 1M
preparation path that `config_n3_1m.yaml` needs landed in that commit, and it is
*newer* than the Gym baked into every image below. A run where the pin didn't
apply is not "slightly older Gym" — it is a different benchmark or no benchmark.

`install_on_the_fly` applies the pin by `git remote add / fetch / checkout` inside
`/opt/Gym`. **That only works if `/opt/Gym` is a git repo**, which differs by
image family:

| Image | `/opt/Gym` | Pin behaviour |
| --- | --- | --- |
| Public `nvcr.io/nvidia/eval-factory/nemo-gym:*` (this template's default) | often **not** a git repo | **Silently ignored** — logs `/opt/Gym is not a git repo; using baked-in Gym version` and runs the baked Gym |
| Internal core-evals gym image (`ci-llm/nemo-gym`, ≥ 2026-07-05 build) | git repo, **build-guarded** | Applies, or **hard-fails** — the build asserts `test -d /opt/Gym/.git`, and the integration layer exits 1 when `HEAD != pin` |

The internal image bakes upstream `NVIDIA-NeMo/Gym` @ `14630a2e` and deliberately
preserves `/opt/Gym/.git` (its CVE-scan `.git` cleanup excludes that one path)
after an earlier build stripped it and broke pinning — the failure mode they
describe is "wrong-but-green eval results."

So on the public image, **verify the pin every run** rather than assuming it:

```bash
grep -c "=== NeMo Gym commit ==="  $RD/logs/client-*.log   # pin applied
grep -c "not a git repo"           $RD/logs/client-*.log   # pin INERT -> baked Gym
```

If the pin is inert, the good case is a loud failure — prepare dies on a missing
`benchmarks/mrcr/config_n3_1m.yaml` because the baked Gym predates the N3
variants. The dangerous case is a baked Gym that has an *older* `config_n3_1m.yaml`
with a different prepare path: it runs green and scores something not comparable
to the golden. If you cannot confirm the `=== NeMo Gym commit ===` SHA, use an
image whose `/opt/Gym` is a git repo instead of trusting the number.
NVIDIA-internal users: `modelopttools:eval-config` Step 3d names one (and how to
pull it); external users should verify the log line on whatever image they have.

## Canary

Unlike GDPVal, MRCR's gym path **does** accept a sample limit in the golden
lineage (`++limit=N`, wired from `limit_samples`). The template omits it; add it
explicitly for a canary rather than assuming the launcher-level
`++…params.limit_samples=N` reaches the gym:

```bash
nel run --config example_gym_mrcr.yaml -o \
  ++evaluation.tasks.0.nemo_evaluator_config.config.params.extra.nemo_gym.collect_rollout_params="<existing string> ++limit=5"
```

Either way, treat the first ~30 minutes of the real run as the canary and check:

```bash
RD=<output_dir>/<run>/nemo_gym.0
grep -c "=== NeMo Gym commit ==="   $RD/logs/client-*.log   # pin actually applied
grep -c "not a git repo"            $RD/logs/client-*.log   # pin inert -> baked Gym
grep -ciE "ModuleNotFoundError|tiktoken" $RD/logs/client-*.log  # pre_cmd didn't take
wc -l $RD/artifacts/evaluator_rollouts.jsonl                # rollouts flowing
```

A run that produces rollouts but scores ~0 across the board almost always means
the **prefix gate** is failing (responses not starting with the required random
prefix) — check a few raw completions before blaming the checkpoint. On a
reasoning model that usually means the reasoning trace is leaking into the graded
answer: verify `--reasoning-parser` is set on the server and
`process_reasoning_traces: true` in the adapter.

## Score Extraction

The headline metric is **`<agent_prefix>/pass@1/accuracy`**, where the prefix is
the variant's agent name (see the variant table). For the 1M template:

```text
mrcr_n3_1m_benchmark_simple_agent/pass@1/accuracy
```

Metrics live in `artifacts/results.yml` (authoritative, local) and are mirrored to
MLflow (prefixed `nemo_gym_`, duplicated under a `key_metrics/` path).

| Metric | Meaning |
| --- | --- |
| `<prefix>/pass@1/accuracy` | **REPORT THIS** — mean prefix-gated SequenceMatcher ratio |
| `<prefix>/pass@k/accuracy` | only meaningful when repeats > 1 |
| `<prefix>/pass@1[avg-of-k]/accuracy` | majority-vote variant, repeats > 1 |
| `<prefix>/n_needles=2\|4\|8/pass@1/accuracy` | per-stratum breakdown — **always quote these too** |

```bash
python3 -c "
import yaml
m=yaml.safe_load(open('<output_dir>/<run>/nemo_gym.0/artifacts/results.yml'))['groups']['nemo_gym']['metrics']
p='mrcr_n3_1m_benchmark_simple_agent'
for k in [f'{p}/pass@1/accuracy'] + [f'{p}/n_needles={n}/pass@1/accuracy' for n in (2,4,8)]:
    if k in m: print(k, '=', m[k]['scores'][k]['value'])"
```

**Always report the needle-count strata alongside the headline.** Long-context
degradation from quantization shows up in the 8-needle stratum first while the
aggregate barely moves — the aggregate alone can hide a real regression.

### Reference point (reviewed golden, BF16 Nemotron Nano 3.5, 1M variant)

```text
mrcr_n3_1m_benchmark_simple_agent/pass@1/accuracy = 26.91
  n_needles=2  36.81
  n_needles=4  27.12
  n_needles=8  16.74
2363/2363 rollouts, 0 failures   (parallelism 256, 4 nodes / 4 instances)
```

Use this to sanity-check a run's shape, not as a pass/fail bar for a different
model. A **rollout count materially below 2363** on the 1M variant means tasks
were lost (e.g. across a walltime resume) and the accuracy is computed over fewer
tasks than the reference — re-check before quoting.
