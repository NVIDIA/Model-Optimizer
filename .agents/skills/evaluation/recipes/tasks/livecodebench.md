# LiveCodeBench v6

## Task Details

- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/nemo_skills.html#nemo-skills-ns-livecodebench>

## Params

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: ns_livecodebench
  container: nvcr.io/nvidia/eval-factory/nemo-skills:26.03
  nemo_evaluator_config:
    config:
      params:
        extra:
          num_repeats: 8
          dataset_split: test_v6_2408_2505
```

## Score Extraction

Result (0-100): `livecodebench_pass_at_1_avg-of-N_accuracy`

N is the repeat count.  If the repeat count is unknown, use the highest available `avg-of-N`.
