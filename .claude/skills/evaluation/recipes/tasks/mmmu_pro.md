# MMMU-Pro

## Task Details

- Task: `ns_mmmu_pro`
- Harness: NeMo Skills, multimodal chat
- Primary metric: `pass@1 symbolic_correct`
- Run time: Medium
- Repeats: 1
- Requires: `HF_TOKEN`
- Reference: https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/nemo_skills.html

## Params

MMMU-Pro is a multimodal task. Use a multimodal-capable endpoint.

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: ns_mmmu_pro
  container: nvcr.io/nvidia/eval-factory/nemo-skills:26.03
  nemo_evaluator_config:
    config:
      params:
        extra:
          num_repeats: 1
```

## Score Extraction

MMMU-Pro accuracy (already in percentage points) comes from:

```text
results.groups."mmmu-pro".metrics.pass@1.scores.symbolic_correct.value
```

`num_repeats: 1` is the standard setting, so `results.yml` does not include
an across-run stderr. The score is computed over a single pass of the
dataset (`stats.count` equals `num_problems`).

```python
import yaml


def extract_mmmu_pro_score(path):
    data = yaml.safe_load(open(path))
    scores = data["results"]["groups"]["mmmu-pro"]["metrics"]["pass@1"]["scores"]
    entry = scores["symbolic_correct"]
    accuracy = entry["value"]
    n = entry.get("stats", {}).get("count")

    return {
        "group": "mmmu-pro",
        "metric": "pass@1",
        "score_key": "symbolic_correct",
        "accuracy": accuracy,
        "stderr": None,
        "n": n,
    }

```
