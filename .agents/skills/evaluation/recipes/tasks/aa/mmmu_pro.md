# MMMU-Pro

## Task Details

- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/index.html>

## Params

MMMU-Pro is a multimodal task. Use a multimodal-capable endpoint.

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: ns_mmmu_pro
  container: nvcr.io/nvidia/eval-factory/nemo-skills:26.03
  nemo_evaluator_config:
    config:
      params:
        extra:
          num_repeats: 1
```

## Score Extraction from mlflow

Result (0-100): `mmmu-pro_pass_at_1_symbolic_correct`
