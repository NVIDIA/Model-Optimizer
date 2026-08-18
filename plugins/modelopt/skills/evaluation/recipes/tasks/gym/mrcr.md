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

Run it through **`"$SKILL_DIR/scripts/nel-gdpval.sh"`**, the same pinned-launcher
wrapper GDPVal uses — the name is GDPVal-flavoured but the pin is gym-wide. This
config forwards `NEL_INVOCATION_ID`, and an unpinned `nel` from PATH can emit the
re-export without first assigning it, exiting on `NEL_INVOCATION_ID: unbound
variable` under `set -u` before the client starts. See `references/gym-gdpval.md`
for the failure signature and the procedure for adopting a newer launcher.

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

The gym bootstrap `command:` block (install_on_the_fly, sub-venv setup, rollout
heredoc, Ray teardown) is **shared with GDPVal**; `references/gym-gdpval.md`
documents that machinery and a fix there applies to both examples.

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

The `num_repeats` column is what each variant **declares upstream**, not
necessarily what runs: for `type: benchmark` datasets the value is a placeholder
and the runner decides. The template therefore pins `++num_repeats=1` in
`common_params`, so **report `pass@1` regardless of variant** unless you raise it
deliberately. **Do not change repeat counts when aligning to a golden.**

### Serving envelope (1M)

- `--max-model-len 1100000` **+** `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` in
  `deployment.env_vars` — vLLM otherwise refuses a len above the checkpoint's
  `max_position_embeddings`.
- `gpu_memory_utilization: 0.95` (vs the usual 0.85) — driven by **sequence
  length**, not prefix caching: a ~1M-token context needs a far larger KV
  allocation, and prefix-cache blocks come out of the same pool.
- `--enable-prefix-caching`, `--enable-chunked-prefill`,
  `--max-num-batched-tokens 131072`.
- **KV dtype: follow the checkpoint, not this template.** Read
  `kv_cache_quant_algo` from the checkpoint's `hf_quant_config.json` and pass the
  matching `--kv-cache-dtype`. vLLM does **not** infer it — `config.json`'s
  `quantization_config` carries no kv_cache key — so an FP8-KV-calibrated
  checkpoint needs the flag explicitly, and a checkpoint without it must **not**
  get `fp8` (that applies uncalibrated KV quantization, worst exactly here where
  error accumulates over ~1M tokens). BF16 KV works and is the safe default, but
  roughly doubles KV footprint, which at 1M is what decides whether the cache
  fits. If baseline and candidate declare different KV algos, that difference is
  part of the delta — report it rather than forcing them equal.
- Fan out via `execution.num_nodes` / `num_instances` (HAProxy pattern A —
  `references/multi-node.md`). **Size these from the cluster's GPUs-per-node**, do
  not copy: pick TP for the model, fill the node with DP, then choose instances for
  the replica count you want. `parallelism` is the total across instances, so
  `--max-num-seqs = ceil(parallelism / num_instances / DP)`. The golden ran 4 nodes
  × (TP2 × DP2) on 4-GPU nodes = 8 replicas, `ceil(256/4/2) = 32` each; the same 8
  replicas on 8-GPU nodes is 2 × (TP2 × DP4).
- **`--max-num-seqs` is a ceiling, not a target.** MRCR is the most KV-bound task
  in the skill — ~1M input tokens per request against AA-LCR's ~120K — so AA-LCR's
  rule applies harder: oversubscribe and vLLM preempts, and recomputing a 1M-token
  prefill makes the run *slower*, not faster. Start small and raise only while
  preemption stays ~0 (`grep -c preempted` the server log). See
  `recipes/tasks/aa/lcr.md` and `references/parallelism.md` ("Balanced sizing").
- **Never cap output.** Answers reproduce a whole earlier turn; a cap truncates it
  and craters the ratio. Golden: `max_new_tokens: null` +
  `++responses_create_params.max_output_tokens=null`.

### Deferred, know the risk

`pre_cmd` installs `tiktoken` / `transformers` **unversioned**, and the n3 prepare
path uses `transformers.AutoTokenizer` to decide which samples exceed the cap — so
a version bump can shift dataset membership even with the Gym pin fixed. The
sub-venv loop also swallows `uv pip install` failures (`|| true`). Both are
inherited from the reviewed golden's `pre_cmd`; pinning would diverge from the run
that produced the reference number. Re-check both if a score moves unexpectedly.

### Gym pin ↔ container — verify before trusting a score

The template pins Gym to `a431501a` (the golden's commit), which carries the N3 1M
prepare path `config_n3_1m.yaml` needs and is **newer than the Gym baked into any
image**. `install_on_the_fly` applies it by `git checkout` in `/opt/Gym`, so it
works only where that is a git repo:

| Image | Pin behaviour |
| --- | --- |
| Public `nvcr.io/nvidia/eval-factory/nemo-gym:*` | **not usable** — `/opt/Gym` is not a git repo, so the bootstrap exits 1 |
| Internal core-evals `ci-llm/nemo-gym` (≥ 2026-07-05) | applies, or **hard-fails** on mismatch |

An inert pin gives either a loud failure (missing `config_n3_1m.yaml`) or — worse
— an older variant that scores green and non-comparable. Verify every run:

```bash
grep -A1 "=== NeMo Gym commit ===" $RD/logs/client-*.log | grep -c a431501a  # pin APPLIED
```

Match the **SHA**, not just the marker — a stale checkout still prints the marker.
The template's `container:` is `???` for this reason and its bootstrap exits 1 if
the pin cannot apply, so only an older copy of this config can score unpinned.

NVIDIA-internal: `modelopttools:eval-config` Step 3d names a working image.

## Canary

MRCR's gym path takes `++limit=N` (the launcher-level `limit_samples` does not
reach the gym). **Not verified on this pinned commit** — treat the first ~30 min
of the real run as the canary, as the GDPVal recipe does. `++limit` caps rollouts
only: the 1M tokenize/drop-over-long **prepare pass still runs in full**, so a
5-sample canary is not cheap. Append it to `collect_rollout_params` in your copy
of the YAML — easier than re-pasting the whole folded scalar through `-o`:

```yaml
                collect_rollout_params: >-
                  ...
                  ++limit=5          # canary only — remove for the scored run
```

Then watch the first ~30 min of the real run:

```bash
RD=<output_dir>/<run>/nemo_gym.0
grep -A1 "=== NeMo Gym commit ===" $RD/logs/client-*.log | grep -c a431501a  # pin applied
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
`process_reasoning_traces: true` in the adapter. That is the **current** adapter
key; `use_reasoning` (used elsewhere in this skill) is its deprecated alias and
the evaluator maps the two to each other, so either works today.

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
rollout count well below 2363 (full runs only — a `++limit` canary is expected
to be short) means tasks were lost (e.g. a walltime resume) and
the score covers fewer tasks than the reference.
