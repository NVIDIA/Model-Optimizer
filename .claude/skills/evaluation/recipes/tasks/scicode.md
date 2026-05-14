# SciCode

SciCode is a NeMo Skills code/reasoning benchmark with multi-step prompts and a
code-execution sandbox. Check this reference before creating or modifying NEL
configs for SciCode; the benchmark has deployment, parallelism, and score
harvesting requirements beyond the task YAML fragment.

## Config Requirements

- Use `--max-model-len 65536` for the deployment. Do not leave the generic
  `32768` fallback in place; SciCode multi-step prompts can exceed 32K tokens.
- Keep `parallelism: 4` unless a canary proves a different value is safe. Higher
  parallelism can flood the code-execution sandbox and produce resource/thread
  failures even when the SLURM job completes.
- Generate enough answer tokens for multi-step solutions:
  `++inference.tokens_to_generate=32768`.
- For reasoning-capable endpoints that support OpenAI-style effort controls, set
  `reasoning_effort: high` through `params_to_add`, not prompt text.
- Use repeats when runtime permits so the result file contains uncertainty
  estimates. The intended full-run plan is `num_repeats: 3`; if using a variant
  that expects `n_repeats`, keep it aligned at `3`. Lower repeat counts are fine
  for canaries, but do not report stderr from a run that did not produce repeat
  statistics.

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: ns_scicode
  nemo_evaluator_config:
    config:
      params:
        max_retries: 10
        parallelism: 4
        extra:
          args: ++inference.tokens_to_generate=32768
          num_repeats: 3
    target:
      api_endpoint:
        adapter_config:
          params_to_remove:
            - max_new_tokens
            - max_completion_tokens
          params_to_add:
            reasoning_effort: high
```

Also make sure the deployment-level args include `--max-model-len 65536`,
preserving any other required model-card or quantization args:

```yaml
deployment:
  extra_args: --max-model-len 65536
```

## Score Extraction

SciCode accuracy comes from:

```text
results.groups.scicode.metrics."pass@1[avg-of-N]".scores.subtask_accuracy.value
```

For repeated runs, report stderr as:

```text
subtask_accuracy_statistics_std_err_across_runs.value * 100 * num_problems / num_subtasks
```

The helper below also supports GPQA's matching layout, where accuracy comes from
`symbolic_correct.value` and stderr is
`symbolic_correct_statistics_std_err_across_runs.value * 100`.

```python
import re
import sys
import yaml


TASKS = {
    "scicode": {
        "score_key": "subtask_accuracy",
        "stderr_scale": "subtasks",
    },
    "gpqa": {
        "score_key": "symbolic_correct",
        "stderr_scale": "percent",
    },
}


def avg_of(metric_name):
    match = re.fullmatch(r"pass@1\[avg-of-(\d+)\]", metric_name)
    return int(match.group(1)) if match else None


def select_pass1_metric(metrics):
    repeated = [name for name in metrics if avg_of(name) is not None]
    if repeated:
        return max(repeated, key=avg_of)
    return "pass@1"


def extract_score(path, group="scicode"):
    spec = TASKS[group]
    data = yaml.safe_load(open(path))
    metrics = data["results"]["groups"][group]["metrics"]
    metric_name = select_pass1_metric(metrics)
    scores = metrics[metric_name]["scores"]

    score_key = spec["score_key"]
    accuracy = scores[score_key]["value"]

    stderr_key = f"{score_key}_statistics_std_err_across_runs"
    stderr_value = scores.get(stderr_key, {}).get("value")
    stderr = None
    if stderr_value is not None:
        if spec["stderr_scale"] == "subtasks":
            num_problems = scores["num_problems"]["value"]
            num_subtasks = scores["num_subtasks"]["value"]
            stderr = stderr_value * 100 * num_problems / num_subtasks
        else:
            stderr = stderr_value * 100

    return {
        "group": group,
        "metric": metric_name,
        "score_key": score_key,
        "accuracy": accuracy,
        "stderr": stderr,
    }


if __name__ == "__main__":
    path = sys.argv[1]
    group = sys.argv[2] if len(sys.argv) > 2 else "scicode"
    print(extract_score(path, group))
```
