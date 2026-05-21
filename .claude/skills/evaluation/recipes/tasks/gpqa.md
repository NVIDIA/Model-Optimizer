# GPQA Diamond

## Task Details

- Run time: Short
- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/nemo_skills.html>

## Params

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: ns_gpqa
  nemo_evaluator_config:
    config:
      params:
        extra:
          args: ++prompt_config=eval/aai/mcq-4choices
          n_samples: 32
    target:
      api_endpoint:
        adapter_config:
          params_to_remove:
            - max_new_tokens
            - max_completion_tokens
```

## Score Extraction

GPQA accuracy comes from:

```text
results.groups.gpqa.metrics."pass@1[avg-of-N]".scores.symbolic_correct.value
```

For repeated runs, report stderr as percentage points:

```text
results.groups.gpqa.metrics."pass@1[avg-of-N]".scores.symbolic_correct_statistics_std_err_across_runs.value * 100
```

Prefer the `pass@1[avg-of-N]` metric matching the configured repeat count. If the
repeat count is unknown, use the highest available `avg-of-N`.
