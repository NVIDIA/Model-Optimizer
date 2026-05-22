# LiveCodeBench v6

## Task Details

- Run time: Medium
- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/nemo_skills.html>

## Params

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: ns_livecodebench
  nemo_evaluator_config:
    config:
      params:
        max_retries: 10
        extra:
          dataset_split: test_v6_2408_2505
          num_repeats: 3
    target:
      api_endpoint:
        adapter_config:
          params_to_remove:
            - max_new_tokens
            - max_completion_tokens
```
