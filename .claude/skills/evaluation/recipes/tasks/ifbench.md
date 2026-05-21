# IFBench

## Task Details

- Run time: Super short
- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/nemo_skills.html>

## Params

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: ns_ifbench
  nemo_evaluator_config:
    config:
      params:
        extra:
          num_repeats: 8
    target:
      api_endpoint:
        adapter_config:
          params_to_remove:
            - max_new_tokens
            - max_completion_tokens
```

## Score Extraction

IFBench primary AA-aligned accuracy (in percentage points) comes from:

```text
results.groups.ifbench.metrics."pass@1[avg-of-N]".scores.prompt_loose_accuracy.value
```

Prefer the `pass@1[avg-of-N]` metric matching the configured repeat count. If the
repeat count is unknown, use the highest available `avg-of-N`.

`results.yml` does **not** include a direct
`prompt_loose_accuracy_statistics_std_err_across_runs`; the closest available
across-run stderr is `prompt_statistics_std_err_across_runs`. It is computed
over the strict + loose prompt-level average rather than
`prompt_loose_accuracy` alone, so report it as an approximate uncertainty.

```text
results.groups.ifbench.metrics."pass@1[avg-of-N]".scores.prompt_statistics_std_err_across_runs.value * 100
```
