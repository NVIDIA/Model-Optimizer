# MMLU-Pro

## Task Details

- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/nemo_skills.html#nemo-skills-ns-mmlu-pro>

## Params

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: ns_mmlu_pro
  nemo_evaluator_config:
    config:
      params:
        extra:
          num_repeats: 1
          args: ++prompt_config=eval/aai/mcq-10choices-boxed
```

## Score Extraction

Result (0-100): `mmlu-pro_pass_at_1_symbolic_correct`
