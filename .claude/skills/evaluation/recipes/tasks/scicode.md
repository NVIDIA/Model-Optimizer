# SciCode

## Task Details

- Run time: Long
- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/nemo_skills.html>

## Params

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
results.groups.scicode.metrics."pass@1[avg-of-N]".scores.subtask_accuracy_statistics_std_err_across_runs.value * 100 * num_problems / num_subtasks
```

Read `num_problems` and `num_subtasks` from the same scores object:

```text
results.groups.scicode.metrics."pass@1[avg-of-N]".scores.num_problems.value
results.groups.scicode.metrics."pass@1[avg-of-N]".scores.num_subtasks.value
```

Prefer the `pass@1[avg-of-N]` metric matching the configured repeat count. If the
repeat count is unknown, use the highest available `avg-of-N`; if no repeated
metric exists, use `pass@1`.
