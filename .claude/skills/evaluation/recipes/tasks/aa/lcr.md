# LCR

## Task Details

- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/nemo_skills.html#nemo-skills-ns-aa-lcr>

## Params

Recommended judge: use Qwen3 235B as an OpenAI-compatible equality-checker
judge, and keep the same judge across comparable runs.

AA-LCR needs long context: plan for roughly 120K input tokens plus 16K
generation tokens. Set deployment `--max-model-len` to at least `131072`, and
use a larger value when the model supports it.

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
  nemo_evaluator_config:
    target:
      api_endpoint:
        adapter_config:
          use_request_logging: false
          use_response_logging: false
    config:
      params:
        extra:
          num_repeats: 16
          judge:
            model_id: <qwen3_235b_judge_model_id>
            url: <openai_compatible_judge_chat_completions_url>
            api_key: INFERENCE_API_KEY
```

## Score Extraction from mlflow

Result (0-100): `aalcr_pass_at_1_avg-of-N_judge_correct`

N is the repeat count.  If the repeat count is unknown, use the highest available `avg-of-N`.
