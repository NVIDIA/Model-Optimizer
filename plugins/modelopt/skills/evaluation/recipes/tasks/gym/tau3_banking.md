# Tau3-Banking (NeMo Gym `tau2` agent, condensed schema)

- Benchmark: <https://github.com/NVIDIA-NeMo/Gym/tree/main/benchmarks/tau2> (`banking_bm25_grep_artificial_analysis`)
- Methodology: [Artificial Analysis Tau3-Banking](https://artificialanalysis.ai/methodology/intelligence-benchmarking#tau3-banking)
- **Source of truth:** `configs/benchmarks/tau3-banking/bench.yaml` + `manifest.yaml` in
  nvidia-eval-factory-benchmarking (`dl/JoC/competitive_evaluation/…`) — re-check before a
  scored run.
- Template: `recipes/examples/gym/example_tau3_banking.yaml`; shared gym machinery:
  `references/gym.md`.

The AA suite's agentic tool-use task, replacing Tau2-Bench Telecom — a different
benchmark, so never compare the two scores. The evaluated model is the banking agent; a
**user simulator** plays the customer. Run shape: 97 `banking_knowledge` tasks × **5**
repeats (485 episodes), BM25 + grep retrieval, GPT-5.4 Mini user (medium reasoning), at
most 200 steps and 10 tool errors per episode. **No LLM judge:** Gym drops `task_102`'s
natural-language assertion, so scores are close to, not identical with, Artificial
Analysis's.

## User simulator (`.env`)

| Key | Example | Notes |
|---|---|---|
| `INFERENCE_API_KEY` | secret | exported; read at run time as `$INFERENCE_API_KEY` |
| `TAU3_USER_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible `/v1` base → `<TAU3_USER_BASE_URL>` |
| `TAU3_USER_MODEL` | `gpt-5.4-mini-2026-03-17` | the endpoint's name for that snapshot → `<TAU3_USER_MODEL>` |

A different user model is a different benchmark — keep it fixed across baseline and
candidate. NVIDIA-internal endpoint: `modelopttools:eval-config` Step 3e.

## Deployment

- **Tool calling is mandatory** (`--enable-auto-tool-choice --tool-call-parser <parser>`),
  or every episode scores 0. Parser from the chat template: XML `<tool_call><function=…>`
  → `qwen3_xml` / `qwen3_coder`; JSON `<tool_call>{"name":…}` → `hermes`. Reasoning
  models: add `--reasoning-parser` and keep thinking on.
- **`--max-model-len` = the checkpoint's full trained context.** The transcript is resent
  every turn and can fill a 262K window (a reviewed run peaked at 262,086); an episode
  that exhausts it ends early on an empty turn, so a smaller cap costs score.
- Carry the model's upstream fragment as a unit (`references/gym.md`), and pass flags
  through `deployment.extra_args` so the launcher keeps `--gpu-memory-utilization`.

## Container and pin

`container: ???` — an image with nemo-evaluator's `nemo_gym` harness, `git`, `uv`, `ray`
(the run-local venv reuses it) and `/usr/local/bin/python3.12`; no `/opt/Gym`.
NVIDIA-internal images: `modelopttools:eval-config` Step 3e.

The template follows canonical `bench.yaml`: Gym `e446e4f4` on Python 3.12. Upstream's
newer pin, Gym `538886bf`, runs Gym on Python 3.13 while tau2 data prep needs 3.12: it
needs a dual-Python image, `runtime_python` and `TAU2_VENV_PYTHON` set to
`/usr/local/bin/python3.13`, and the data pins
`NEMO_GYM_TAU2_BENCH_DATA_REPO_URL=https://github.com/bxyu-nvidia/tau2-bench`,
`NEMO_GYM_TAU2_BENCH_DATA_REF=7245a8f6f12a6046e3443b120428dcb695fe1f6e`. The pin is part
of the benchmark build — keep it fixed across compared runs.

## Parallelism

`parallelism` is Gym's total client concurrency. The reviewed runs use the canonical
`2048`, which admits all 485 episodes at once, so the user simulator sees the full episode
rate. On a rate-limited endpoint, canary first and lower it until the client log shows no
429s — throttled user turns fail episodes and depress the score.

## Canary

`-o ++evaluation.nemo_evaluator_config.config.params.limit_samples=5` (the bootstrap passes
`--limit`); data prep still runs in full.

```bash
RD=<output_dir>/<run>/nemo_gym.0
grep -A1 "=== NeMo Gym commit ===" $RD/logs/client-*.log | grep -c e446e4f4      # pin applied
grep -ciE "429|rate limit|openai_api_key" $RD/logs/client-*.log                 # user-sim auth / throttling
wc -l $RD/artifacts/evaluator_rollouts.jsonl                                     # rollouts flowing
grep -cE '"(function_call|tool_calls)"' $RD/artifacts/evaluator_rollouts.jsonl   # 0 = wrong/missing parser
```

## Score Extraction

Report `tau2_banking_knowledge_bm25_grep_artificial_analysis_agent/mean/reward` (0–1;
MLflow prefixes `nemo_gym_`) over **485** rollouts (`…/banking_knowledge/num_samples_total`;
fewer means lost episodes). Quote with it, same prefix:
`trajectory_termination_reason/empty_tool_calls_and_content/pct` (context exhaustion),
`message_finish_reason/length/pct`, `trajectory_transfer_to_human_agents/pct`.

Reviewed runs span about 0.07–0.45 across models; repeat runs of one model agree within
±0.02, so treat a baseline-vs-candidate delta inside that band as noise.
