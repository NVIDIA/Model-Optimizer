# Slurm configuration

Use the runner file for site-wide Slurm settings and the execution file for
stage-specific choices.

## Partitions and logs

`runner.slurm.partition` sets the default for stages without a partition
override. It accepts one partition name or a list of eligible names. Omit it to
use the site's Slurm default. A stage can set
`execution.stages.<stage>.partition` to one name or its own eligible list.

`runner.slurm.log_dir` sets the directory used for every attempt log, including
the final-report attempt. When omitted, logs are written below
`<puzzle_dir>/logs`. Relative values are resolved from `puzzle_dir`; absolute
paths are used as written.

The runner loader accepts `partition_interactive`, `partition_batch`,
`partition_cpu`, and `interactive_max_nodes` as compatibility fields. They
infer stage routing from role names and node count, which assumes a particular
site layout and duplicates execution-stage settings. Maintained configs and
examples use `runner.slurm.partition` with stage overrides instead.

The production examples also avoid literal `interactive` and `batch` stage
overrides because those partition names are not portable between Slurm sites.

## CPU-only stages

Set `resource: cpu` with a CPU partition override for work that does not need a
GPU. Other stages continue to use the runner default:

```yaml
runner:
  kind: slurm
  slurm:
    partition:
      - gpu-general
      - gpu-overflow

execution:
  stages:
    convert:
      strategy: single
      resource: cpu
      partition:
        - cpu-general
        - cpu-overflow
    width_importance:
      strategy: single
```

Slurm selects one partition from each eligible list. The
[`runner.slurm.example.yaml`](../configs/orchestration/runner.slurm.example.yaml)
and [`execution.example.yaml`](../configs/orchestration/execution.example.yaml)
files show the runner default and per-stage CPU routing together. The CPU-only
`final_report` task accepts only a `partition` override.

## Scheduler settings and model settings

Do not put `sequence_parallel` under
`execution.stages.<stage>.parallel`. That mapping controls scheduler allocation
and accepts mesh dimensions such as `tp`, `pp`, and `dp_replicate`.
`sequence_parallel` changes model execution and belongs in the experiment's
model-parallel profile. Setup-generated execution files omit it for this
reason.

Runner and execution files reject unknown fields and suggest the closest valid
name when possible.
