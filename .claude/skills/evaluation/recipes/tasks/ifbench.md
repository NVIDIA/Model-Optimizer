# IFBench

## Task Details

- Task: `ns_ifbench`
- Harness: NeMo Skills, chat
- Primary metric: `pass@1[avg-of-8] prompt_loose_accuracy`
- Run time: Super short
- Repeats: 8

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

IFBench accuracy comes from:

```text
results.groups.ifbench.metrics."pass@1[avg-of-N]".scores.prompt_loose_accuracy.value
```
