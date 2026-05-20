# HLE AA

## Task Details

- Task: `ns_hle_aa`
- Harness: nemo-skills, chat
- Primary metric: `pass@1 judge_correct`
- Run time: Long
- Repeats: 1
- Requires: `HF_TOKEN`, `JUDGE_API_KEY`
- Reference: https://docs.nvidia.com/nemo/evaluator/nightly/evaluation/benchmarks/catalog/all/harnesses/nemo_skills.html

## Params

This is the text-only HLE task with params aligned to Artificial Analysis Index
v2. HLE is judge-scored and requires judge credentials.

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: ns_hle_aa
  container: nvcr.io/nvidia/eval-factory/nemo-skills:26.03
  env_vars:
    HF_TOKEN: host:HF_TOKEN
    JUDGE_API_KEY: host:JUDGE_API_KEY
  nemo_evaluator_config:
    config:
      params:
        extra:
          judge:
            model_id: <hle_aa_judge_model_id>
            url: <openai_compatible_judge_chat_completions_url>
            api_key: JUDGE_API_KEY
```

## Score Extraction

HLE AA accuracy comes from:

```text
results.groups.hle.metrics.pass@1.scores.judge_correct.value
```

```python
import yaml


def extract_ns_hle_aa_score(path):
    data = yaml.safe_load(open(path))
    scores = data["results"]["groups"]["hle"]["metrics"]["pass@1"]["scores"]
    accuracy = scores["judge_correct"]["value"]
    symbolic = scores.get("symbolic_correct", {}).get("value")
    n = scores["judge_correct"].get("stats", {}).get("count")

    return {
        "group": "hle",
        "metric": "pass@1",
        "score_key": "judge_correct",
        "accuracy": accuracy,
        "symbolic_correct": symbolic,
        "stderr": None,
        "n": n,
    }

```
