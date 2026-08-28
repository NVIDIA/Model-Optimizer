# HLE

## Task Details

- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/nemo_skills.html#nemo-skills-ns-hle-aa>

## Params

Text-only HLE, params aligned to Artificial Analysis Index v2; judge-scored.
Substitute the judge `model_id`/`url` with the literal values you keep in `.env`
(`HLE_JUDGE_MODEL_ID` rec. **GPT-4o**, `NS_JUDGE_URL`; see `recipes/env.example`),
they're config, not secrets, so they don't need exporting. Only `api_key`
(`INFERENCE_API_KEY`) is exported and read by the harness. Keep the judge fixed
across comparable runs.

## Contract Source

Select `ns_hle_aa` through `examples/llm_eval/nel_config.py`, supplying its
documented judge model ID and `/v1` endpoint options. The task definition and
credential wiring live only in `examples/llm_eval/task_contracts.yaml`.

## Score Extraction from mlflow

Result (0-100): `hle_pass_at_1_judge_correct`
