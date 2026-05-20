# MMLU-Pro

## Task Details

- Task: `nemo_skills.ns_mmlu_pro`
- Harness: NeMo Skills, chat
- Primary metric: `pass@1 symbolic_correct`
- Run time: Short
- Repeats: 1
- Requires: None
- Reference: https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/nemo_skills.html

## Params

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: nemo_skills.ns_mmlu_pro
  nemo_evaluator_config:
    config:
      params:
        extra:
          num_repeats: 1
          args: ++prompt_config=eval/aai/mcq-10choices-boxed ++inference.tokens_to_generate=null
    target:
      api_endpoint:
        adapter_config:
          params_to_remove:
            - max_new_tokens
            - max_completion_tokens
```

## Score Extraction

```text
results.groups.mmlu_pro.metrics.pass@1.scores.symbolic_correct.value
```

`num_repeats: 1` is the standard setting, so `results.yml` does not include
an across-run stderr. The score is computed over a single pass of the
dataset (`stats.count` equals `num_problems`).

```python
import yaml


def extract_mmlu_pro_score(path):
    data = yaml.safe_load(open(path))
    scores = data["results"]["groups"]["mmlu_pro"]["metrics"]["pass@1"]["scores"]
    entry = scores["symbolic_correct"]
    accuracy = entry["value"]
    n = entry.get("stats", {}).get("count")

    return {
        "group": "mmlu_pro",
        "metric": "pass@1",
        "score_key": "symbolic_correct",
        "accuracy": accuracy,
        "stderr": None,
        "n": n,
    }
```
