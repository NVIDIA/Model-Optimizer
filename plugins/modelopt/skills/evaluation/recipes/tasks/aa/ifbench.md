# IFBench

## Task Details

- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/index.html>

## Params

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: ns_ifbench
  container: nvcr.io/nvidia/eval-factory/nemo-skills:26.03
  nemo_evaluator_config:
    config:
      params:
        extra:
          num_repeats: 5
```

## Score Extraction from mlflow

Result (0-100): `ifbench_pass_at_1_avg-of-N_prompt_loose_accuracy`

N is the repeat count.  If the repeat count is unknown, use the highest available `avg-of-N`.
