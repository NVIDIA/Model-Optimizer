# GPQA Diamond

## Task Details

- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/simple_evals.html#simple-evals-gpqa-diamond-aa-v3>

## Params

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: gpqa_diamond_aa_v3
  container: nvcr.io/nvidia/eval-factory/simple-evals:26.03
  nemo_evaluator_config:
    config:
      params:
        extra:
          n_samples: 16
```

## Score Extraction from mlflow

Result (0-100): `gpqa_diamond_score_micro_avg_of_N`

N is the repeat count.  If the repeat count is unknown, use the highest available `avg_of_N`.
