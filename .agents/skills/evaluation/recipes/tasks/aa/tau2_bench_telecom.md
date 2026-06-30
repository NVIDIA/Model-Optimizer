# Tau2 Bench Telecom

## Task Details

- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/tau2_bench.html#tau2-bench-tau2-bench-telecom>

## Params

Tau2 uses the evaluated model as the agent plus a separate user-simulator endpoint;
keep both fixed across runs. The judger (**gpt-oss-120B**) and user-simulator
(**Qwen3 235B**) `model_id`s are hardcoded in the fragment below — swap them for
equivalents on your own endpoint if needed. Only the shared `url`
(`TAU2_ENDPOINT_URL`) comes from `.env` (see `recipes/env.example`) — config, not a
secret, so no export needed; only `api_key` (`INFERENCE_API_KEY`) is exported.
tau2-bench needs the full `/v1/chat/completions` URL (nemo-skills judges use the
`/v1` base).

For parallelism, we have to throttle to a smaller cap due to the test may be throttled by
user and judger API rate limit. If frequent 429 errors are hit, the reported scores could be much lower.

The `parallelism:` field is left as `???` — the right value depends on the
judge and user-simulator endpoints' rate limits, which vary per deployment.
Start with a conservative canary value (e.g. 32–128), watch the logs for 429
errors from the judger/user endpoints, and ramp up if stable. The hard
upper bound is 512. After choosing a value, recompute the deployment's
`--max-num-seqs` per the rule in SKILL.md Step 3.

## Deployment requirement — tool calling (mandatory)

Tau2 is agentic tool use, so the served model MUST launch with
`--enable-auto-tool-choice --tool-call-parser <parser>` — without it every trial
returns `avg_reward 0.0` with zero `tool_calls`. Pick `<parser>` from the
checkpoint's `chat_template.jinja`: XML `<tool_call><function=…>` → `qwen3_coder`
(Qwen3/3.5); JSON `<tool_call>{"name":…}` → `hermes`.

> With `--reasoning-parser` on, some turns emit only `<think>…</think>` and no
> tool call → empty-response failures that depress the score; if frequent,
> disable thinking for Tau2 (`enable_thinking: false`).

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: tau2_bench_telecom
  container: nvcr.io/nvidia/eval-factory/tau2-bench:26.03
  env_vars:
    INFERENCE_API_KEY: host:INFERENCE_API_KEY
  nemo_evaluator_config:
    config:
      params:
        parallelism: ???   # required: see body above; cap 512; recompute --max-num-seqs after setting
        extra:
          cache:
            cache_dir: /results/native_cache
            enabled: true
          skip_failed_samples: true
          n_samples: 8
          user:
            model_id: nvidia/qwen/qwen-235b    # Qwen3 235B; use an equivalent on your own endpoint if needed
            url: <TAU2_ENDPOINT_URL>           # from .env (full /v1/chat/completions)
            api_key: INFERENCE_API_KEY         # env-var name; exported, read by harness
          judger:
            model_id: nvidia/openai/gpt-oss-120b   # gpt-oss-120B; use an equivalent on your own endpoint if needed
            url: <TAU2_ENDPOINT_URL>           # from .env (full /v1/chat/completions)
            api_key: INFERENCE_API_KEY         # env-var name; exported, read by harness
```

## Score Extraction

Result (0-1): `tau2_bench_telecom_pass_at_1`
