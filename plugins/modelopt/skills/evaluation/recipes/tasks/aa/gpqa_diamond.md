# GPQA Diamond

## Task Details

- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/index.html>

## Params

nemo-skills `ns_gpqa` on the AA 4-choice MCQ prompt
(`++prompt_config=eval/aai/mcq-4choices`), `num_repeats: 16`.

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: ns_gpqa
  container: nvcr.io/nvidia/eval-factory/nemo-skills:26.03
  nemo_evaluator_config:
    config:
      params:
        extra:
          num_repeats: 16
          args: "++prompt_config=eval/aai/mcq-4choices"
```

## Score Extraction from mlflow

Result (0-100): `gpqa_pass_at_1_avg-of-N_symbolic_correct`

N is the repeat count (16 here, so `gpqa_pass_at_1_avg-of-16_symbolic_correct`).
If the repeat count is unknown, use the highest available `avg-of-N`.
