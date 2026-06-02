# SciCode

## Task Details

- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/nemo_skills.html#nemo-skills-ns-scicode>

## Params

SciCode is a NeMo Skills code/reasoning benchmark with multi-step prompts and a
code-execution sandbox. Check this reference before creating or modifying NEL
configs for SciCode; the benchmark has deployment, parallelism, and score
harvesting requirements beyond the task YAML fragment.

## Config Requirements

- **Deployment context length:** at least `--max-model-len 65536` (SciCode
  multi-step prompts can exceed 32K). The example template's default of
  `--max-model-len 131072` satisfies this and is preferred — do not lower
  it unless you have a memory reason to.

## YAML Fragment

Use this inside the top-level `evaluation.tasks` list:

```yaml
- name: ns_scicode
  container: nvcr.io/nvidia/eval-factory/nemo-skills:26.03
  nemo_evaluator_config:
    config:
      params:
        extra:
          args: ++prompt_config=eval/scicode/default ++with_background=true
          num_repeats: 8
```

## Score Extraction from mlflow

Result (0-100): `scicode_pass_at_1_avg-of-N_subtask_accuracy`

N is the repeat count.  If the repeat count is unknown, use the highest available `avg-of-N`.
