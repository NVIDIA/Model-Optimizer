# Tau2 Bench Telecom

## Task Details

- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/tau2_bench.html#tau2-bench-tau2-bench-telecom>

## Params

Tau2 Bench uses the evaluated model as the agent and a separate LLM endpoint as
the user simulator. Configure the user simulator explicitly and keep it fixed
across comparable runs.

For parallelism, we have to throttle to a smaller cap due to the test may be throttled by
user and judger API rate limit. If frequent 429 errors are hit, the reported scores could be much lower.

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
        parallelism: capped at 512 or maybe smaller
        extra:
          cache:
            cache_dir: /results/native_cache
            enabled: true
          skip_failed_samples: true
          n_samples: 8
          user:
            model_id: <user_simulator_qwen_235b_model_id>
            url: <openai_compatible_user_simulator_chat_completions_url>
            api_key: INFERENCE_API_KEY
          judger:
            model_id: <judger_gpt_oss_120b_model_id>
            url: <openai_compatible_user_simulator_chat_completions_url>
            api_key: INFERENCE_API_KEY
```

## Score Extraction

Result (0-1): `tau2_bench_telecom_pass_at_1`
