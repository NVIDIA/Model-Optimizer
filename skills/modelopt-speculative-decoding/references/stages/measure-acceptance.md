# Stage 5 — Measure acceptance length with AIPerf

Measure a drafter's acceptance length (AL) in a running vLLM server: vLLM reports
speculative-decoding counters per request, and [AIPerf](https://github.com/ai-dynamo/aiperf)
replays a benchmark dataset and saves them in each request record. Works for any
drafter vLLM serves: a pipeline export, a published checkpoint, an MTP head.

The result is **pooled AL**, `1 + sum(accepted draft tokens) / sum(verification steps)`
over the run — not `specdec_bench`'s `Average_AL` (see the last section).

## Prerequisites

- vLLM 0.29.0 or later (`--per-request-spec-decode-metrics`); validated on 0.33.0.
- AIPerf at commit `d1f80b6`; the 0.13.0 release has no built-in SPEED-Bench datasets:
  `pip install "aiperf @ git+https://github.com/ai-dynamo/aiperf@d1f80b6f04dd3b14f1c5afd955ef1ec7d26612a7"`
- SPEED-Bench only: a Hugging Face token whose account accepted the gated `cais/hle` terms.

| `--public-dataset` | `--num-conversations` | `--request-count` |
| --- | --- | --- |
| `speed_bench_qualitative` | 880 | 1138 (185 sessions are multi-turn) |
| `spec_al_gsm8k` | 1319 | 1319 |
| `spec_al_mtbench` | 80 | 160 |

For `speed_bench_throughput_32k`, count both from the resolved file (Step 1).

## Step 1 — Resolve the dataset before taking GPUs

SPEED-Bench publishes placeholders for prompts it cannot redistribute, and AIPerf
fetches them from the source datasets on first use. That takes minutes, or over an
hour on a parallel filesystem, so set `HF_HOME` to local disk or to a cache that
already holds those datasets. AIPerf caches under `.cache/aiperf/datasets/`
**relative to the current directory**: run every `aiperf` command from the directory
used here.

```bash
python -c "from aiperf.dataset.loader.speed_bench_public import SpeedBenchPublicLoader as L; print(L.resolve_config('qualitative'))"
python -c "
import collections, json
c = collections.Counter()
for line in open('.cache/aiperf/datasets/speed-bench/qualitative.jsonl'):
    r = json.loads(line); c[r['category']] += len(r['messages'])  # one request per turn
print(dict(sorted(c.items())), sum(c.values()))"
```

## Step 2 — Serve

```bash
vllm serve <target> \
  --speculative-config '{"method": "eagle3", "model": "<drafter>", "num_speculative_tokens": 7}' \
  --per-request-spec-decode-metrics detailed \
  --max-model-len 16384 --max-num-seqs 224 --max-num-batched-tokens 2048 \
  --no-enable-prefix-caching --seed 0
```

- `detailed` adds the per-step arrays needed for per-position acceptance curves.
- Pass the `--speculative-config` JSON as a separate argument (no `=`): the publisher
  in Step 5 parses the saved command.
- No reasoning parser, so the streamed `content` that AIPerf replays as the next turn's
  history carries the whole reply, as in `specdec_bench`.
- The last two lines reproduce `specdec_bench`'s in-process engine at concurrency 32,
  7 draft tokens and `--max_seq_len 16384` (`max_num_seqs` = concurrency x draft
  tokens). Keep them when the result will sit next to `specdec_bench` numbers, and
  check that both logs report the same `GPU KV cache size`.

## Step 3 — Run AIPerf

```bash
aiperf profile --model <served-model-name> --tokenizer <target> \
  --url http://127.0.0.1:8000 --endpoint-type chat --streaming \
  --public-dataset speed_bench_qualitative --dataset-sampling-strategy sequential \
  --num-conversations 880 --request-count 1138 --concurrency 32 \
  --extra-inputs '{"temperature": 0, "top_p": 1.0, "top_k": 0, "max_tokens": 4096, "ignore_eos": false,
                  "chat_template_kwargs": {"enable_thinking": true}}' \
  --use-server-token-count --no-server-metrics --no-gpu-telemetry --ui-type none \
  --export-level records --artifact-dir <run-dir>
```

- Set **both** run bounds. `--request-count` alone starts extra sessions and cuts
  multi-turn ones short; `--num-conversations` alone keeps sessions whole but gives the
  run no request budget, which the publisher rejects.
- Keep `ignore_eos` false and set no `min_tokens`: forcing the output length lowers AL.
- Send `temperature`, `top_p` and `top_k` explicitly; `vllm serve` fills unset fields
  from the model's `generation_config.json`.
- Always set the reasoning mode explicitly in `chat_template_kwargs`, even to the
  model's default: it changes AL, and template defaults differ by model. Use the keys
  the model's chat template reads (`reasoning_effort`, `enable_thinking`, ...).
- Unset `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` for the client: offline mode makes
  this AIPerf build fail to load the tokenizer, even from a local path.
- Smoke test: SPEED-Bench rows are grouped by category, so the first K sessions of the
  sequential sampler cover one category. Use `--dataset-sampling-strategy shuffle
  --random-seed <s>`, derive the bounds from AIPerf's own sampler (below), and label
  the run `coverage: subset`.

```python
import collections, json
from aiperf.common import random_generator as rng
from aiperf.dataset.dataset_samplers import ShuffleSampler

rows = [json.loads(l) for l in open(".cache/aiperf/datasets/speed-bench/qualitative.jsonl")]
rng.init(1)                                                 # --random-seed
sampler = ShuffleSampler([r["question_id"] for r in rows])  # file order, as AIPerf passes it
plan = {sampler.next_conversation_id() for _ in range(33)}  # --num-conversations
counts = collections.Counter()
for r in rows:
    if r["question_id"] in plan:
        counts[r["category"]] += len(r["messages"])
print(dict(sorted(counts.items())), sum(counts.values()))   # sum = --request-count
```

## Step 4 — Read the AL

```python
import json
acc = steps = 0
for line in open("<run-dir>/profile_export.jsonl"):
    r = json.loads(line)
    if r["metadata"].get("benchmark_phase") == "profiling":
        acc += r["spec_decode_acceptance"]["num_accepted_draft_tokens"]
        steps += r["spec_decode_acceptance"]["num_spec_steps"]
print(1 + acc / steps)
```

This is `spec_decode_token_weighted_acceptance_length` in `profile_export_aiperf.json`;
its `spec_decode_acceptance_length` is a mean over requests. Per-category AL: group by
`metadata.source_kind`. To group the turns of one conversation use
`metadata.x_correlation_id` — `session_num` is assigned per request. A record without
`spec_decode_acceptance` is missing data, not zero.

## Step 5 — Publish to the team tracker (NVIDIA internal)

The team's internal speculative-decoding dashboard takes these runs from its
publisher, `acceptance_results.py` (ask the speculative-decoding team where it lives).
It validates the run against a JSON manifest, then writes one MLflow run with the AL,
per-category AL, positional curves and the raw AIPerf files.

```bash
python acceptance_results.py <run-dir> --manifest manifest.json --output acceptance.json            # validate
python acceptance_results.py <run-dir> --manifest manifest.json --output acceptance.json --publish  # publish
```

Fill the manifest from the run as executed: `name`, `profile` (`speed-qualitative`,
`speed-throughput-32k`, `gsm8k` or `mt-bench`, matching `--public-dataset`), `coverage`
(`full` or `subset`), `dataset_revision`, `expected_categories` (requests per category
after multi-turn expansion, Step 1), `model`, `model_revision`, `draft_model`,
`draft_revision`, `method`, `draft_length`, `draft_topology` (`linear` enables the
positional curves), `backend: vllm`, `runtime_version`, `aiperf_revision`,
`btk_revision` (required; say "not used" when AIPerf runs directly), `gpu`,
`gpu_count`, `concurrency`, `output_tokens`, `temperature`, `acceptance_mode: real`,
`routing: real`, `ignore_eos: false`, `chat_template_args` (exactly the `chat_template_kwargs`
sent), and the exact `server_command` and `client_command`.

- `draft_model` must equal the `model` in `--speculative-config`. To serve a local
  copy under its Hub id, start `vllm serve` in a directory holding it at `<org>/<repo>`.
- No credentials in either command (`--api-key`, tokens).
- The publisher must read AIPerf `d1f80b6` exports: dataset provenance under
  `metadata.dataset`, sessions keyed by `x_correlation_id`. One that does not fails
  with "Saved AIPerf dataset provenance does not match the manifest", or with "MT-Bench
  is missing a conversation turn" on a complete MT-Bench run.
- Always validate before `--publish`, which writes to the shared experiment the
  dashboard reads; test uploads do not belong there.

## Compare with specdec_bench

`specdec_bench` counts AL over streamed output chunks, and each turn's first chunk
(the prefill token) counts as a step. Match the definition before comparing:

- Pooled: `1 + sum(Joint_Acceptance_Rate[k] for k in 2..D+1)` from its
  `acceptance_rate.json` is the counterpart of Step 4; the prefill chunks move it by
  about 0.1%.
- Per conversation: `Average_AL` and `Category_AL` average over conversations. The
  AIPerf counterpart is `sum(completion_tokens) / sum(num_spec_steps + 1)` over each
  conversation's turns, averaged over conversations. It can differ from the pooled AL
  by several percent in either direction.

Greedy decoding in vLLM is not batch-invariant: temperature-0 replicates, even on
servers started with identical flags, differ by up to about 1% overall and a few
percent per category. Repeat a run before reading anything into a gap that size.
