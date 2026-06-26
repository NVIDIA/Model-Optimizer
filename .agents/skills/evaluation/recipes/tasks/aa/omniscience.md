# AA-Omniscience

## Task Details

- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/nemo_skills.html#nemo-skills-ns-omniscience>

## Params

Knowledge / hallucination benchmark, params aligned to Artificial Analysis Index
v2; judge-scored. The judge `model_id` is hardcoded in the fragment below
(**gcp/google/gemini-3-flash-preview**) — swap it for an equivalent on your own
endpoint if needed. The judge `url` comes from `.env` (`INFERENCE_JUDGE_URL`); only
`api_key` (`INFERENCE_API_KEY`) is exported and read by the harness. Keep the judge
fixed across comparable runs.

`++parse_reasoning=False` is required (golden knob — omniscience scores the final
answer, not the reasoning trace).

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: nemo_skills.ns_omniscience
  container: nvcr.io/nvidia/eval-factory/nemo-skills:26.05.1
  env_vars:
    INFERENCE_API_KEY: host:INFERENCE_API_KEY
  nemo_evaluator_config:
    config:
      params:
        extra:
          num_repeats: 10
          args: "++parse_reasoning=False"
          judge:
            api_key: INFERENCE_API_KEY
            model_id: gcp/google/gemini-3-flash-preview   # use an equivalent on your own endpoint if needed
            url: <INFERENCE_JUDGE_URL>                     # from .env (/v1 base)
```

## Score Extraction from mlflow

**Primary result — Omniscience Index** (-100 to 100): `omniscience_pass_at_1_avg-of-N_judge_omni_index`. This is AA's headline metric: accuracy net of hallucinations, rewarding abstention over guessing wrong (so it can be negative). Report this one. See the AA methodology: <https://artificialanalysis.ai/methodology/intelligence-benchmarking#aa-omniscience>.

Also report (same `pass_at_1_avg-of-N` aggregation):

- **Accuracy** (0-100): `omniscience_pass_at_1_avg-of-N_judge_correct` — % of questions answered correctly.
- **Non-hallucination rate** (0-100): `100 - omniscience_pass_at_1_avg-of-N_judge_omni_hallucination` — the `judge_omni_hallucination` key is the hallucination rate, so non-hallucination = `1 - hallucination`.

N is the repeat count (10). If the repeat count is unknown, use the highest available `avg-of-N`.
