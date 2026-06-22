# HLE

## Task Details

- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/nemo_skills.html#nemo-skills-ns-hle-aa>

## Params

Text-only HLE, params aligned to Artificial Analysis Index v2; judge-scored.
Substitute the judge `model_id`/`url` with the literal values you keep in `.env`
(`HLE_JUDGE_MODEL_ID` rec. **GPT-4o**, `NS_JUDGE_URL`; see `recipes/env.example`) —
they're config, not secrets, so they don't need exporting. Only `api_key`
(`INFERENCE_API_KEY`) is exported and read by the harness. Keep the judge fixed
across comparable runs.

`hle_strict_judge: true` (inside the `judge` block) enables strict judging.

**Don't add `++server.enable_soft_fail=True` for a self-deployed model.** In
nemo-skills 0.7.0 it forces the client to load a tokenizer from the served model
name, which 404s when that name isn't a real HF repo (failing the run). If you
need soft-fail, also set `++tokenizer` to an HF id or a container-loadable local
tokenizer.

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
            model_id: <HLE_JUDGE_MODEL_ID>   # from .env; recommended GPT-4o
            url: <NS_JUDGE_URL>              # from .env (/v1 base)
            api_key: INFERENCE_API_KEY       # env-var name; exported, read by harness
            hle_strict_judge: true
```

## Score Extraction from mlflow

Result (0-100): `hle_pass_at_1_judge_correct`
