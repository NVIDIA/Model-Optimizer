# GPQA Diamond

## Task Details

- Task: `ns_gpqa`
- Harness: NeMo Skills, chat
- Primary metric: `pass@1[avg-of-16] symbolic_correct`
- Run time: Short
- Samples: 16
- Requires: None
- Reference: https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/nemo_skills.html

## Params

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: ns_gpqa
  nemo_evaluator_config:
    config:
      params:
        extra:
          args: ++prompt_config=eval/aai/mcq-4choices
          n_samples: 16
    target:
      api_endpoint:
        adapter_config:
          params_to_remove:
            - max_new_tokens
            - max_completion_tokens
```

## Score Extraction

GPQA accuracy comes from:

```text
results.groups.gpqa.metrics."pass@1[avg-of-N]".scores.symbolic_correct.value
```

For repeated runs, report stderr as percentage points:

```text
symbolic_correct_statistics_std_err_across_runs.value * 100
```

Prefer the `pass@1[avg-of-N]` metric matching the configured repeat count. If the
repeat count is unknown, use the highest available `avg-of-N`.

```python
import re
import yaml


def avg_of(metric_name):
    match = re.fullmatch(r"pass@1\[avg-of-(\d+)\]", metric_name)
    return int(match.group(1)) if match else None


def select_metric(metrics, repeats=None):
    if repeats is not None:
        expected = f"pass@1[avg-of-{repeats}]"
        if expected in metrics:
            return expected

    repeated = [name for name in metrics if avg_of(name) is not None]
    if repeated:
        return max(repeated, key=avg_of)
    return "pass@1"


def extract_gpqa_score(path, repeats=None):
    data = yaml.safe_load(open(path))
    metrics = data["results"]["groups"]["gpqa"]["metrics"]
    metric_name = select_metric(metrics, repeats)
    scores = metrics[metric_name]["scores"]

    accuracy = scores["symbolic_correct"]["value"]
    stderr_value = scores.get(
        "symbolic_correct_statistics_std_err_across_runs", {}
    ).get("value")
    stderr = stderr_value * 100 if stderr_value is not None else None

    return {
        "group": "gpqa",
        "metric": metric_name,
        "score_key": "symbolic_correct",
        "accuracy": accuracy,
        "stderr": stderr,
    }

```
