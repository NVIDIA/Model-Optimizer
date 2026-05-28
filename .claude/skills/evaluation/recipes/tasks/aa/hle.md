# HLE

## Task Details

- Reference: <https://docs.nvidia.com/nemo/evaluator/nightly/evaluation/benchmarks/catalog/all/harnesses/nemo_skills.html#nemo-skills-ns-hle-aa>

## Params

This is the text-only HLE task with params aligned to Artificial Analysis Index
v2. HLE is judge-scored and requires judge credentials.

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: ns_hle_aa
  container: nvcr.io/nvidia/eval-factory/nemo-skills:26.03
  env_vars:
    INFERENCE_API_KEY: host:INFERENCE_API_KEY
  nemo_evaluator_config:
    config:
      params:
        extra:
          judge:
            model_id: <hle_aa_judge_model_id>
            url: <openai_compatible_judge_chat_completions_url>
            api_key: INFERENCE_API_KEY
```

## Score Extraction from mlflow

Result (0-100): `hle_pass_at_1_judge_correct`
