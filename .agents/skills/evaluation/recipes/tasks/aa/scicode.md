# SciCode

## Task Details

- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/nemo_skills.html#nemo-skills-ns-scicode>

## Params

SciCode is a NeMo Skills code/reasoning benchmark with multi-step prompts and a
code-execution sandbox. Check this reference before creating or modifying NEL
configs for SciCode; the benchmark has deployment, parallelism, and
score-harvesting requirements beyond its task contract.

## Config Requirements

- **Deployment context length:** at least `--max-model-len 65536` (SciCode
  multi-step prompts can exceed 32K). The example template's larger default
  satisfies this; do not lower it without verifying the workload still fits.

## Contract Source

Select `ns_scicode` through `examples/llm_eval/nel_config.py`. Its container,
prompt settings, and repeat count live only in
`examples/llm_eval/task_contracts.yaml`.

## Score Extraction from mlflow

Result (0-100): `scicode_pass_at_1_avg-of-N_subtask_accuracy`

N is the repeat count.  If the repeat count is unknown, use the highest available `avg-of-N`.
