# GPQA Diamond

## Task Details

- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/simple_evals.html#simple-evals-gpqa-diamond-aa-v3>

## Params

## Contract Source

Select `gpqa_diamond_aa_v3` through `examples/llm_eval/nel_config.py`. Its
container and sample count are defined only in
`examples/llm_eval/task_contracts.yaml`.

## Score Extraction from mlflow

Result (0-100): `gpqa_diamond_score_micro_avg_of_N`

N is the repeat count.  If the repeat count is unknown, use the highest available `avg_of_N`.
