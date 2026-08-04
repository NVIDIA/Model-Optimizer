# MMLU-Pro

## Task Details

- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/index.html>

## Params

nemo-skills `ns_mmlu_pro` on the AA 10-choice boxed MCQ prompt
(`++prompt_config=eval/aai/mcq-10choices-boxed`), `num_repeats: 1`.

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: ns_mmlu_pro
  container: nvcr.io/nvidia/eval-factory/nemo-skills:26.03
  nemo_evaluator_config:
    config:
      params:
        extra:
          num_repeats: 1
          args: "++prompt_config=eval/aai/mcq-10choices-boxed"  # pragma: allowlist secret
```

## Score Extraction from mlflow

Result (0-100): `mmlu-pro_pass_at_1_symbolic_correct`  (note the hyphen in `mmlu-pro`)
