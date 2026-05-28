# GPQA Diamond

## Task Details

- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/nemo_skills.html#nemo-skills-ns-gpqa>

## Params

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: ns_gpqa
  container: nvcr.io/nvidia/eval-factory/nemo-skills:26.03
  nemo_evaluator_config:
    config:
      params:
        extra:
          args: ++prompt_config=eval/aai/mcq-4choices
          num_repeats: 16
```

## Score Extraction from mlflow

Result (0-100): `gpqa_pass_at_1_avg-of-N_symbolic_correct`

N is the repeat count.  If the repeat count is unknown, use the highest available `avg-of-N`.
