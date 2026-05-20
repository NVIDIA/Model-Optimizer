# AA-LCR

## Task Details

- Task: `aa_lcr`
- Harness: AA-LCR, chat
- Primary metric: `accuracy.accuracy`
- Run time: Long
- Samples: 3
- Requires: `HF_TOKEN`, `JUDGE_API_KEY`
- Reference: https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/AA-LCR.html

## Params

Recommended judge: use Qwen3 235B as an OpenAI-compatible equality-checker
judge, and keep the same judge across comparable runs.

AA-LCR is long-context sensitive. For 128K-context models, avoid capping
generation tokens for this task unless the deployment needs the cap for
stability.

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: aa_lcr
  container: nvcr.io/nvidia/eval-factory/aa-lcr:26.03
  env_vars:
    HF_TOKEN: host:HF_TOKEN
    JUDGE_API_KEY: host:JUDGE_API_KEY
  nemo_evaluator_config:
    config:
      params:
        extra:
          n_samples: 3
          judge:
            model_id: <qwen3_235b_judge_model_id>
            url: <openai_compatible_judge_chat_completions_url>
            api_key: JUDGE_API_KEY
```

## Score Extraction

AA-LCR accuracy comes from:

```text
results.groups.aa_lcr.metrics.accuracy.scores.accuracy.value
results.groups.aa_lcr.metrics.accuracy.scores.accuracy.stats.stderr
```

```python
import yaml


def extract_aa_lcr_score(path):
    data = yaml.safe_load(open(path))
    scores = data["results"]["groups"]["aa_lcr"]["metrics"]["accuracy"]["scores"]
    entry = scores["accuracy"]
    accuracy = entry["value"] * 100
    stderr = entry.get("stats", {}).get("stderr")
    stderr_pp = stderr * 100 if stderr is not None else None

    return {
        "group": "aa_lcr",
        "metric": "accuracy",
        "score_key": "accuracy",
        "accuracy": accuracy,
        "stderr": stderr_pp,
    }

```
