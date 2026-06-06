# LCR

## Task Details

- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/nemo_skills.html#nemo-skills-ns-aa-lcr>

## Params

Judge-scored (equality checker). Substitute the judge `model_id`/`url` with the
literal values you keep in `.env` (`LCR_JUDGE_MODEL_ID` rec. **Qwen3 235B**,
`NS_JUDGE_URL`; see `recipes/env.example`) — config, not secrets, so no export
needed; only `api_key` (`INFERENCE_API_KEY`) is exported. Keep the judge fixed.

AA-LCR needs long context: plan for roughly 120K input tokens plus 16K
generation tokens. Set deployment `--max-model-len` to at least `131072`, and
use a larger value when the model supports it.

**Parallelism — set this *lower* than the top-level default.** AA-LCR is the
suite's most concurrency-sensitive task on two fronts at once. (1) *KV-bound:* each
request carries ~120K input tokens, so its KV footprint is large and a high
`parallelism` triggers preemption — and recomputing 120K-token prefills is hugely
wasteful, so over-parallelizing here makes the run *slower*, not faster (see
`references/parallelism.md`, "Balanced sizing"). (2) *Judge-bound:* the
equality-checker endpoint rate-limits before your served model does. So give it an
explicit per-task `parallelism` well below the model/GPU-bound tasks' value: start
small (≈16–32 for GQA models; MLA models such as Kimi tolerate several× more) and
raise only while preemption ≈ 0 and the judge shows no 429s. The field is left as
`???`; after choosing a value, recompute the deployment's `--max-num-seqs` per
SKILL.md Step 3 (sized off the *max* parallelism across all tasks).

## YAML Fragment

LCR has a deployment-side requirement (`--max-model-len 131072`) and a task
block. Per SKILL.md Step 3, the deployment flag must live inside
`deployment.command:` — not in the deprecated `extra_args` field.

**Deployment requirement:** ensure the `vllm serve ...` invocation in
`deployment.command` includes `--max-model-len 131072` (or higher).

```yaml
- name: ns_aa_lcr
  container: nvcr.io/nvidia/eval-factory/nemo-skills:26.03
  env_vars:
    INFERENCE_API_KEY: host:INFERENCE_API_KEY
    LOG_LEVEL: lit:WARNING # Skip logging the long context inputs.
  nemo_evaluator_config:
    target:
      api_endpoint:
        adapter_config:
          use_request_logging: false
          use_response_logging: false
    config:
      params:
        parallelism: ???   # set LOWER than top-level: long-context (KV-bound) + judge-bound; see body above. Recompute --max-num-seqs after setting.
        extra:
          num_repeats: 16
          judge:
            model_id: <LCR_JUDGE_MODEL_ID>   # from .env; recommended Qwen3 235B
            url: <NS_JUDGE_URL>              # from .env (/v1 base)
            api_key: INFERENCE_API_KEY       # env-var name; exported, read by harness
```

## Score Extraction from mlflow

Result (0-100): `aalcr_pass_at_1_avg-of-N_judge_correct`

N is the repeat count.  If the repeat count is unknown, use the highest available `avg-of-N`.
