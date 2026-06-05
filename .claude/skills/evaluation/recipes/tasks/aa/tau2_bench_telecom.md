# Tau2 Bench Telecom

## Task Details

- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/tau2_bench.html#tau2-bench-tau2-bench-telecom>

## Params

Tau2 uses the evaluated model as the agent plus a separate user-simulator endpoint;
keep both fixed across runs. Substitute the user-sim & judger `model_id`/`url` with the
literal values you keep in `.env` (`TAU2_USER_MODEL_ID` rec. **Qwen3 235B**,
`TAU2_JUDGER_MODEL_ID` rec. **gpt-oss-120B**, `TAU2_ENDPOINT_URL`; see
`recipes/env.example`) — config, not secrets, so no export needed; only `api_key`
(`INFERENCE_API_KEY`) is exported. tau2-bench needs the full `/v1/chat/completions`
URL (nemo-skills judges use the `/v1` base).

For parallelism, we have to throttle to a smaller cap due to the test may be throttled by
user and judger API rate limit. If frequent 429 errors are hit, the reported scores could be much lower.

The `parallelism:` field is left as `???` — the right value depends on the
judge and user-simulator endpoints' rate limits, which vary per deployment.
Start with a conservative canary value (e.g. 32–128), watch the logs for 429
errors from the judger/user endpoints, and ramp up if stable. The hard
upper bound is 512. After choosing a value, recompute the deployment's
`--max-num-seqs` per the rule in SKILL.md Step 3.

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
            model_id: <TAU2_USER_MODEL_ID>     # from .env; recommended Qwen3 235B
            url: <TAU2_ENDPOINT_URL>           # from .env (full /v1/chat/completions)
            api_key: INFERENCE_API_KEY         # env-var name; exported, read by harness
          judger:
            model_id: <TAU2_JUDGER_MODEL_ID>   # from .env; recommended gpt-oss-120B
            url: <TAU2_ENDPOINT_URL>           # from .env (full /v1/chat/completions)
            api_key: INFERENCE_API_KEY         # env-var name; exported, read by harness
```

## Score Extraction

Result (0-1): `tau2_bench_telecom_pass_at_1`
