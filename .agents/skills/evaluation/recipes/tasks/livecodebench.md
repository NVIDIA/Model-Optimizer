# LiveCodeBench v6

## Task Details

- Reference: <https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/nemo_skills.html#nemo-skills-ns-livecodebench>

## Params

## Contract Source

Select `ns_livecodebench` through `examples/llm_eval/nel_config.py`. Its
container, repeat count, and pinned dataset split live only in
`examples/llm_eval/task_contracts.yaml`.

## Score Extraction

Result (0-100): `livecodebench_pass_at_1_avg-of-N_accuracy`

N is the repeat count.  If the repeat count is unknown, use the highest available `avg-of-N`.
