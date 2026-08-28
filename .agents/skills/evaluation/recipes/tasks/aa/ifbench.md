# IFBench

## Task Details

- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/nemo_skills.html#nemo-skills-ns-ifbench>

## Params

## Contract Source

Select `ns_ifbench` through `examples/llm_eval/nel_config.py`. Its container and
repeat count live only in `examples/llm_eval/task_contracts.yaml`.

## Score Extraction from mlflow

Result (0-100): `ifbench_pass_at_1_avg-of-N_prompt_loose_accuracy`

N is the repeat count.  If the repeat count is unknown, use the highest available `avg-of-N`.
