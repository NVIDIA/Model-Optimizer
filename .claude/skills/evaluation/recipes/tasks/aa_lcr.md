# AA-LCR

## Task Details

- Task: `nemo_skills.ns_aa_lcr`
- Harness: NeMo Skills, chat
- Primary metric: `pass@1 judge_correct`
- Run time: Long
- Repeats: 3
- Requires `HF_TOKEN` and `JUDGE_API_KEY`

Recommended judge: use Qwen3 235B A22B 2507 Non-Reasoning as an
OpenAI-compatible equality-checker judge, and keep the same judge across
comparable runs.

AA-LCR is long-context sensitive. For 128K-context models, avoid capping
generation tokens for this task unless the deployment needs the cap for
stability.

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: nemo_skills.ns_aa_lcr
  container: nvcr.io/nvidia/eval-factory/nemo-skills:26.03
  env_vars:
    HF_TOKEN: host:HF_TOKEN
    JUDGE_API_KEY: host:JUDGE_API_KEY
  nemo_evaluator_config:
    target:
      api_endpoint:
        adapter_config:
          params_to_remove:
            - max_tokens
    config:
      params:
        extra:
          num_repeats: 3
          judge:
            model_id: <qwen3_235b_a22b_2507_non_reasoning_judge_model_id>
            url: <openai_compatible_judge_url>
            api_key: JUDGE_API_KEY
```

## Score Extraction

AA-LCR accuracy comes from:

```text
results.groups.aalcr.metrics.pass@1.scores.judge_correct.value
```
