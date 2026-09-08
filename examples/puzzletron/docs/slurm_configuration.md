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

## Stage resource defaults

Puzzletron defaults data preparation, checkpoint conversion, library building,
MIP solving, filtering, and reporting to `resource: cpu`. Model execution stages
default to `resource: gpu`. These values control the scheduler request, not
where the orchestration command itself runs.

An explicit stage resource overrides the default. This is useful when running
locally on an interactive GPU node or when a site uses GPU nodes for CPU work.
Choosing `resource: cpu` for code that actually calls CUDA leaves GPUs hidden and
can fail at runtime; choosing `resource: gpu` for CPU work reserves a GPU that the
stage may not use.

Set a CPU partition override when CPU work should use a separate Slurm queue:

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

For CPU stages, `runner.slurm.cpu_cpus_per_task` sets the requested CPU count and
`runner.slurm.cpu_memory_mb` sets memory in MiB. Omit them to use the site's Slurm
defaults. These settings are part of resume identity, so changing either causes
Puzzletron to submit the stage with the new allocation instead of treating an
older active attempt as the same work.

## Scheduler settings and model settings

Do not put `sequence_parallel` under
`execution.stages.<stage>.parallel`. That mapping controls scheduler allocation
and accepts mesh dimensions such as `tp`, `pp`, and `dp_replicate`.
`sequence_parallel` changes model execution and belongs in the experiment's
model-parallel profile. Setup-generated execution files omit it for this
reason.

Runner and execution files reject unknown fields and suggest the closest valid
name when possible.

## Worker setup hooks

`runner.execution_contract.prerun_commands` and `postrun_commands` are copied
into generated worker scripts and appear in dry-run output. Puzzletron rejects
obvious literal assignments to credential-like variables so those values are
not persisted. Inherit credentials from the launch environment, require an
existing variable such as `${API_KEY:?set API_KEY}`, retrieve it from a secret
command, or source a permission-protected `setup_env` file. This check catches
common mistakes but is not a shell parser or a complete credential scanner.
