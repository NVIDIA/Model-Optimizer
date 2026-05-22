# Tau2 Bench Telecom

## Task Details

- Run time: Long
- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/tau2_bench.html#tau2-bench-tau2-bench-telecom>

## Params

Tau2 Bench uses the evaluated model as the agent and a separate LLM endpoint as
the user simulator. Configure the user simulator explicitly and keep it fixed
across comparable runs.

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: tau2_bench_telecom
  container: nvcr.io/nvidia/eval-factory/tau2-bench:26.03
  env_vars:
    USER_API_KEY: host:USER_API_KEY
  nemo_evaluator_config:
    config:
      params:
        extra:
          user:
            model_id: <user_simulator_model_id>
            url: <openai_compatible_user_simulator_chat_completions_url>
            api_key: USER_API_KEY
```

## Score Extraction

TODO
