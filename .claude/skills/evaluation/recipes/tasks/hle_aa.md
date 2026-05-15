# HLE AA

## Task Details

- Task: `nemo_skills.ns_hle_aa`
- Harness: NeMo Skills, chat
- Primary metric: `pass@1 judge_correct`
- Run time: Long
- Repeats: 1
- Requires `HF_TOKEN` and `INFERENCE_API_KEY`

Recommended judge: use OpenAI GPT-4o as the OpenAI-compatible equality-checker
judge, matching the original HLE paper setup, and keep the same judge across
comparable runs.

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: nemo_skills.ns_hle_aa
  container: nvcr.io/nvidia/eval-factory/nemo-skills:26.03
  env_vars:
    HF_TOKEN: host:HF_TOKEN
    INFERENCE_API_KEY: host:INFERENCE_API_KEY
  nemo_evaluator_config:
    config:
      params:
        extra:
          judge:
            model_id: <gpt_4o_judge_model_id>
            api_key: INFERENCE_API_KEY
```

## Score Extraction

HLE AA accuracy comes from:

```text
results.groups.hle.metrics.pass@1.scores.judge_correct.value
```
