# IFBench

## Task Details

- Task: `ns_ifbench`
- Harness: NeMo Skills, chat
- Primary metric: `pass@1[avg-of-8] prompt_loose_accuracy`
- Run time: Super short
- Repeats: 8
- Requires: None
- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/nemo_skills.html>

## Params

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: ns_ifbench
  nemo_evaluator_config:
    config:
      params:
        extra:
          num_repeats: 8
    target:
      api_endpoint:
        adapter_config:
          params_to_remove:
            - max_new_tokens
            - max_completion_tokens
```

## Score Extraction

IFBench primary AA-aligned accuracy (in percentage points) comes from:

```text
results.groups.ifbench.metrics."pass@1[avg-of-N]".scores.prompt_loose_accuracy.value
```

`results.yml` does **not** include a direct
`prompt_loose_accuracy_statistics_std_err_across_runs`; the closest available
across-run stderr is `prompt_statistics_std_err_across_runs`. It is computed
over the strict + loose prompt-level average rather than
`prompt_loose_accuracy` alone, so report it as an approximate uncertainty.

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


def extract_ifbench_score(path, repeats=None):
    data = yaml.safe_load(open(path))
    metrics = data["results"]["groups"]["ifbench"]["metrics"]
    metric_name = select_metric(metrics, repeats)
    scores = metrics[metric_name]["scores"]

    accuracy = scores["prompt_loose_accuracy"]["value"]
    proxy_stderr_value = scores.get(
        "prompt_statistics_std_err_across_runs", {}
    ).get("value")
    stderr = proxy_stderr_value * 100 if proxy_stderr_value is not None else None

    return {
        "group": "ifbench",
        "metric": metric_name,
        "score_key": "prompt_loose_accuracy",
        "accuracy": accuracy,
        "stderr": stderr,
        "stderr_source": "prompt_statistics_std_err_across_runs (proxy)",
    }

```
