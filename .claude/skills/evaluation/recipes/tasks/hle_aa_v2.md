# HLE AA v2

## Task Details

- Task: `hle_aa_v2`
- Harness: HLE, chat
- Primary metric: `pass@1 judge_correct`
- Run time: Long
- Repeats: 1
- Requires: `HF_TOKEN`, `JUDGE_API_KEY`
- Reference: https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/hle.html

## Params

This is the text-only HLE task with params aligned to Artificial Analysis Index
v2. HLE is judge-scored and requires judge credentials.

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: hle_aa_v2
  container: nvcr.io/nvidia/eval-factory/hle:26.03
  env_vars:
    HF_TOKEN: host:HF_TOKEN
    JUDGE_API_KEY: host:JUDGE_API_KEY
```

## Score Extraction

HLE AA v2 accuracy comes from:

```text
results.groups.hle.metrics.pass@1.scores.judge_correct.value
```
