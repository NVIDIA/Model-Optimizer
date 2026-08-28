# LCR

## Task Details

- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/nemo_skills.html#nemo-skills-ns-aa-lcr>

## Params

Judge-scored (equality checker). Substitute the judge `model_id`/`url` with the
literal values you keep in `.env` (`LCR_JUDGE_MODEL_ID` rec. **Qwen3 235B**,
`NS_JUDGE_URL`; see `recipes/env.example`), config, not secrets, so no export
needed; only `api_key` (`INFERENCE_API_KEY`) is exported. Keep the judge fixed.

AA-LCR needs long context: plan for roughly 120K input tokens plus generation
tokens. Set deployment `--max-model-len` to at least `131072`, and use a larger
value when the model and expected output length require it.

**Parallelism, set this *lower* than the top-level default.** AA-LCR is the
suite's most concurrency-sensitive task on two fronts at once. (1) *KV-bound:* each
request carries ~120K input tokens, so its KV footprint is large and a high
`parallelism` triggers preemption, and recomputing 120K-token prefills is hugely
wasteful, so over-parallelizing here makes the run *slower*, not faster (see
`references/parallelism.md`, "Balanced sizing"). (2) *Judge-bound:* the
equality-checker endpoint rate-limits before your served model does. So give it an
explicit per-task `parallelism` well below the model/GPU-bound tasks' value: start
small (≈16–32 for GQA models; MLA models such as Kimi tolerate several× more) and
raise only while preemption ≈ 0 and the judge shows no 429s. The field is left as
`???`; after choosing a value, recompute the deployment's `--max-num-seqs` per
SKILL.md Step 3 (sized off the *max* parallelism across all tasks).

## Contract Source

Select `ns_aa_lcr` through `examples/llm_eval/nel_config.py`, supplying the
judge model ID, `/v1` endpoint, and conservative task parallelism. The task
definition, safe logging settings, repeat count, and credential wiring live
only in `examples/llm_eval/task_contracts.yaml`.

Keep the deployment requirement in `deployment.command`, not in the task
contract or the deprecated `extra_args` field.

## Score Extraction from mlflow

Result (0-100): `aalcr_pass_at_1_avg-of-N_judge_correct`

N is the repeat count.  If the repeat count is unknown, use the highest available `avg-of-N`.
