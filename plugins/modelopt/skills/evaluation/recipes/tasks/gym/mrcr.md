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
`pre_cmd`), plus `NEMO_EVALUATOR_TRUST_UNLISTED_TASKS=1` — `nemo_gym` is not in
the FDF mapping, so submission is refused without it.

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

**Preempted vs timed out.** A 1M run routinely exceeds 4h. `TIMEOUT` auto-resumes
from the response cache; `CANCELLED by <uid>` (preemption) does not — its chained
job exits in ~20s (`…finished with 'CANCELLED…' state. EXIT!`), which is expected,
not a bug. Resume by hand: `cd <run>/nemo_gym.0 && sbatch run.sub`. Progress is
cumulative — check `wc -l evaluator_rollouts.jsonl` before assuming loss.

Rollouts flowing but scores ~0 = the **prefix gate** failing, not a bad
checkpoint. On a reasoning model that is usually the reasoning trace leaking into
the graded answer — check `--reasoning-parser` on the server and
`process_reasoning_traces: true` in the adapter.

## Score Extraction

**Not in `results.yml`** — its `groups.nemo_gym.metrics` map is empty for MRCR
(only `key_metrics/mean/*` token telemetry). Read
`artifacts/evaluator_rollouts_aggregate_metrics.json` → `[0].agent_metrics`:

| Key | |
| --- | --- |
| `pass@1/accuracy` | **REPORT THIS** — already 0-100, do not ×100 |
| `n_needles=2\|4\|8/pass@1/accuracy` | per-stratum — **always quote too** |
| `mean/reward` | same number as a 0-1 fraction (= `mean/seq_match_ratio`) |
| `mean/prefix_matched` | prefix-gate pass rate; **~0.55 is healthy** |

```bash
python3 -c "
import json
m=json.load(open('<output_dir>/<run>/nemo_gym.0/artifacts/evaluator_rollouts_aggregate_metrics.json'))[0]['agent_metrics']
print('pass@1', round(m['pass@1/accuracy'],2))
for n in (2,4,8): print(f'  n={n}', round(m[f'n_needles={n}/pass@1/accuracy'],2))
print('prefix_matched', round(m['mean/prefix_matched'],3))"
```

Quantization damage hits the **8-needle stratum first** while the aggregate stays
flat. Before quoting, check truncation: `eval_factory_metrics.json` →
`response_stats.finish_reason.length` (a capped response scores ~0; measured 2.4%
at `--max-model-len 1100000`).

Reference shape (reviewed golden, BF16 Nano 3.5, 1M): `pass@1 = 26.91` (2/4/8
needles = 36.81 / 27.12 / 16.74), 2363/2363 rollouts, parallelism 256, 4 nodes /
4 instances. Use it to sanity-check shape, not as a bar for another model — a
rollout count well below 2363 means tasks were lost (e.g. a walltime resume) and
the score covers fewer tasks than the reference.
