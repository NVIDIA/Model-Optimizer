# LCR

## Task Details

- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/nemo_skills.html#nemo-skills-ns-aa-lcr>

## Params

Judge-scored (equality checker). The judge `model_id` is hardcoded in the fragment
below (**Qwen3 235B**) — swap it for an equivalent on your own endpoint if needed.
The judge `url` still comes from `.env` (`INFERENCE_JUDGE_URL`; see `recipes/env.example`)
— config, not a secret, so no export; only `api_key` (`INFERENCE_API_KEY`) is
exported. Keep the judge fixed.

AA-LCR needs long context. The longest prompts are ~120K **input** tokens, and
the **generation** budget adds on top of that — ~16K for non-reasoning models but
**up to ~64K for reasoning models**. `--max-model-len` must cover
**input + max generation**, so the old `131072` floor is **too low**: 120K + 64K
≈ 185K overflows it, and every AA-LCR request then fails with vLLM HTTP 400
(`maximum context length` / `VLLMValidationError`) — no generations reach the
judge, so the whole task fails with no score (this hit **5 of 9 day0 models**).

Set `--max-model-len` to **`max(163840, longest_input + max_new_tokens)`**, capped
at the model's `max_position_embeddings` (working day0 values were 163840–262144).
Use the **same value on both the baseline and the quantized deployment**.

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

LCR has a deployment-side requirement (a large `--max-model-len`, see above) and
a task block. Per SKILL.md Step 3, the deployment flag must live inside
`deployment.command:` — not in the deprecated `extra_args` field.

**Deployment requirement:** ensure the `vllm serve ...` invocation in
`deployment.command` includes `--max-model-len` set to
`max(163840, longest_input + max_new_tokens)` (capped at the model's
`max_position_embeddings`) — **not** the old `131072`, which overflows for
reasoning models. Match the value on baseline and quantized deployments.

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
            model_id: nvidia/qwen/qwen-235b   # Qwen3 235B; use an equivalent on your own endpoint if needed
            url: <INFERENCE_JUDGE_URL>              # from .env (/v1 base)
            api_key: INFERENCE_API_KEY       # env-var name; exported, read by harness
```

## Score Extraction from mlflow

Result (0-100): `aalcr_pass_at_1_avg-of-N_judge_correct`

N is the repeat count.  If the repeat count is unknown, use the highest available `avg-of-N`.
