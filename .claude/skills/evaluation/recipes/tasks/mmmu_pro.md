# MMMU-Pro

## Task Details

- Run time: Medium
- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/nemo_skills.html>

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
    target:
      api_endpoint:
        adapter_config:
          params_to_remove:
            - max_new_tokens
            - max_completion_tokens
```

## Score Extraction

MMMU-Pro accuracy (already in percentage points) comes from:

```text
results.groups."mmmu-pro".metrics.pass@1.scores.symbolic_correct.value
```

`num_repeats: 1` is the standard setting, so `results.yml` does not include
an across-run stderr. The score is computed over a single pass of the
dataset. The judged item count is:

```text
results.groups."mmmu-pro".metrics.pass@1.scores.symbolic_correct.stats.count
```
