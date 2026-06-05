# MMLU-Pro

## Task Details

- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/simple_evals.html#simple-evals-mmlu-pro-aa-v3>

## Params

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: mmlu_pro_aa_v3
  container: nvcr.io/nvidia/eval-factory/simple-evals:26.03
```

## Score Extraction

Result (0-1): `mmlu_pro_score_micro`
