# HLE AA

## Task Details

- Run time: Long
- Reference: <https://docs.nvidia.com/nemo/evaluator/nightly/evaluation/benchmarks/catalog/all/harnesses/nemo_skills.html>

## Params

This is the text-only HLE task with params aligned to Artificial Analysis Index
v2. HLE is judge-scored and requires judge credentials.

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: ns_hle_aa
  container: nvcr.io/nvidia/eval-factory/nemo-skills:26.03
  env_vars:
    HF_TOKEN: host:HF_TOKEN
    JUDGE_API_KEY: host:JUDGE_API_KEY
  nemo_evaluator_config:
    config:
      params:
        extra:
          judge:
            model_id: <hle_aa_judge_model_id>
            url: <openai_compatible_judge_chat_completions_url>
            api_key: JUDGE_API_KEY
```

## Score Extraction

HLE AA accuracy comes from:

```text
results.groups.hle.metrics.pass@1.scores.judge_correct.value
```

HLE also exposes symbolic extraction accuracy for sanity checks:

```text
results.groups.hle.metrics.pass@1.scores.symbolic_correct.value
```

With `num_repeats: 1`, `results.yml` does not include an across-run stderr. The
judged item count is:

```text
results.groups.hle.metrics.pass@1.scores.judge_correct.stats.count
```
