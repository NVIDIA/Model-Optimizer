# MRCR (OpenAI Multi-Round Co-reference Resolution, NeMo Gym `simple_agent`)

## Task Details

- Benchmark: <https://github.com/NVIDIA-NeMo/Gym/tree/main/benchmarks/mrcr>
- Resource server: <https://github.com/NVIDIA-NeMo/Gym/blob/main/resources_servers/mrcr/configs/mrcr.yaml>
- Dataset: `openai/mrcr` (HF, gated → `HF_TOKEN`)

Long-context retrieval. Each task is a long multi-turn conversation with N
near-identical "needle" responses; the model must reproduce the Nth verbatim
behind a random prefix. Deterministic scoring: `SequenceMatcher.ratio()`, **0
unless the response starts with the required prefix**. Stratified by needle count
(2/4/8); accuracy falls sharply as N rises.

A 0.2.6 `nel` `nemo_gym` task (not nel-next), so Steps 1–9 apply. **Standalone** —
one gym eval per config, never mixed with other tasks.

**Not an AA benchmark** — never generate it for an "AA" request. It shares
`recipes/tasks/gym/` with GDPVal, which *is* AA: the dir groups by **harness**,
not suite, so read membership per task.

Much lighter than GDPVal: `simple_agent`, **no SIF sandbox, no judge, no Tavily** —
`HF_TOKEN` is the only secret, and the cost is context length rather than agent
turns. Like GDPVal it needs `NEMO_EVALUATOR_TRUST_PRE_CMD=1` (the config has a
`pre_cmd`).

## Config

Start from the self-contained example — do **not** copy fragments into another
config:

```text
recipes/examples/gym/example_mrcr.yaml   # SLURM + vLLM, 1M variant
```

### Variant — pick first

Sets the context cap, the dataset, **and the metric prefix**. The golden uses 1M.

| Gym config | Cap (tokenizer) | Metric prefix | `num_repeats` |
| --- | --- | --- | --- |
| `benchmarks/mrcr/config_n3_1m.yaml` | 1,048,576 (gated NVIDIA) | `mrcr_n3_1m_benchmark_simple_agent` | 1 |
| `benchmarks/mrcr/config_n3_128k.yaml` | 131,072 (gated NVIDIA) | `mrcr_n3_128k_benchmark_simple_agent` | 1 |
| `benchmarks/mrcr/config.yaml` | none (`o200k_base`) | `mrcr_benchmark_simple_agent` | 4 |

The n3 variants drop over-long samples, so all three are different datasets and
**not comparable to each other**. Pick one, keep it fixed across baseline and
candidate, and set it in **both** `data_prep_params` and `collect_rollout_params`
— changing one prepares one dataset and rolls out another.

`num_repeats` comes from the variant; the template does not override it (1M
reports `pass@1`). Upstream it is a placeholder for `type: benchmark` datasets —
the real count comes from the runner. **Do not change repeat counts when aligning
to a golden.**

### Serving envelope (1M)

- `--max-model-len 1100000` **+** `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` in
  `deployment.env_vars` — vLLM otherwise refuses a len above the checkpoint's
  `max_position_embeddings`.
- `gpu_memory_utilization: 0.95` (vs the usual 0.85) for the KV cache.
- `--enable-prefix-caching`, `--enable-chunked-prefill`,
  `--max-num-batched-tokens 131072`.
- `--kv-cache-dtype fp8` — **itself a precision choice**; keep identical across
  baseline and candidate or the delta also measures KV-cache quantization.
- Fan out via `execution.num_nodes` / `num_instances` (golden **4 / 4**, HAProxy
  pattern A — `references/multi-node.md`). `parallelism` is the total across
  instances, so `--max-num-seqs = ceil(parallelism / num_instances / DP)`
  (256/4/1 = 64).
- **Never cap output.** Answers reproduce a whole earlier turn; a cap truncates it
  and craters the ratio. Golden: `max_new_tokens: null` +
  `++responses_create_params.max_output_tokens=null`.

### Gym pin ↔ container — verify before trusting a score

The template pins Gym to `a431501a` (the golden's commit), which carries the N3 1M
prepare path `config_n3_1m.yaml` needs and is **newer than the Gym baked into any
image**. `install_on_the_fly` applies it by `git checkout` in `/opt/Gym`, so it
works only where that is a git repo:

| Image | Pin behaviour |
| --- | --- |
| Public `nvcr.io/nvidia/eval-factory/nemo-gym:*` (template default) | often **silently ignored** — logs `/opt/Gym is not a git repo`, runs baked Gym |
| Internal core-evals `ci-llm/nemo-gym` (≥ 2026-07-05) | applies, or **hard-fails** on mismatch |

An inert pin gives either a loud failure (missing `config_n3_1m.yaml`) or — worse
— an older variant that scores green and non-comparable. Verify every run:

```bash
grep -c "=== NeMo Gym commit ==="  $RD/logs/client-*.log   # applied
grep -c "not a git repo"           $RD/logs/client-*.log   # INERT
```

NVIDIA-internal: `modelopttools:eval-config` Step 3d names a working image.

## Canary

MRCR's gym path accepts `++limit=N` (unlike GDPVal). Append it explicitly — the
launcher-level `limit_samples` does not reach the gym:

```bash
nel run --config example_mrcr.yaml -o \
  ++evaluation.tasks.0.nemo_evaluator_config.config.params.extra.nemo_gym.collect_rollout_params="<existing> ++limit=5"
```

Then watch the first ~30 min of the real run:

```bash
RD=<output_dir>/<run>/nemo_gym.0
grep -c "=== NeMo Gym commit ==="        $RD/logs/client-*.log   # pin applied
grep -ciE "ModuleNotFoundError|tiktoken" $RD/logs/client-*.log   # pre_cmd didn't take
wc -l $RD/artifacts/evaluator_rollouts.jsonl                     # rollouts flowing
```

Rollouts flowing but scores ~0 = the **prefix gate** failing, not a bad
checkpoint. On a reasoning model that is usually the reasoning trace leaking into
the graded answer — check `--reasoning-parser` on the server and
`process_reasoning_traces: true` in the adapter.

## Score Extraction

Headline: **`<prefix>/pass@1/accuracy`** (prefix per the variant table). Metrics
are in `artifacts/results.yml`, mirrored to MLflow as `nemo_gym_…` plus a
`key_metrics/` copy.

| Metric | |
| --- | --- |
| `<prefix>/pass@1/accuracy` | **REPORT THIS** — mean prefix-gated ratio |
| `<prefix>/n_needles=2\|4\|8/pass@1/accuracy` | per-stratum — **always quote too** |
| `<prefix>/pass@k/…`, `<prefix>/pass@1[avg-of-k]/…` | only meaningful when repeats > 1 |

```bash
python3 -c "
import yaml
m=yaml.safe_load(open('<output_dir>/<run>/nemo_gym.0/artifacts/results.yml'))['groups']['nemo_gym']['metrics']
p='mrcr_n3_1m_benchmark_simple_agent'
for k in [f'{p}/pass@1/accuracy'] + [f'{p}/n_needles={n}/pass@1/accuracy' for n in (2,4,8)]:
    if k in m: print(k, '=', m[k]['scores'][k]['value'])"
```

Quantization damage shows in the **8-needle stratum first** while the aggregate
barely moves — the aggregate alone hides it.

Reference shape (reviewed golden, BF16 Nano 3.5, 1M): `pass@1 = 26.91` (2/4/8
needles = 36.81 / 27.12 / 16.74), 2363/2363 rollouts, parallelism 256, 4 nodes /
4 instances. Use it to sanity-check shape, not as a bar for another model — a
rollout count well below 2363 means tasks were lost (e.g. a walltime resume) and
the score covers fewer tasks than the reference.
