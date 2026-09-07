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

The runner loader accepts `partition_interactive`, `partition_batch`, and
`interactive_max_nodes` as compatibility fields. They infer stage routing from
role names and node count, which assumes a particular site layout and duplicates
execution-stage settings. Maintained configs use `runner.slurm.partition`, the
supported `partition_cpu` fallback, and stage overrides instead.

The production examples also avoid literal `interactive` and `batch` stage
overrides because those partition names are not portable between Slurm sites.

## Stage resource defaults

Puzzletron defaults data preparation, checkpoint conversion, library building,
MIP solving, filtering, and reporting to `resource: cpu`. Model execution stages
default to `resource: gpu`. These values control the scheduler request, not
where the orchestration command itself runs.

Named MIP configurations use the solve-only CPU driver. Checkpoint
materialization and validation belong to explicit post-MIP stages. Legacy MIP
configurations without named runs use GPUs when in-stage realization validation
is enabled, taking their mesh from `realize_model.automodel.parallel` with
`replacement_scoring.automodel.parallel` as fallback. Contradictory MIP resource
overrides are rejected with the execution setting to change.

An explicit stage resource overrides the default. This is useful when running
locally on an interactive GPU node or when a site uses GPU nodes for CPU work.
Choosing `resource: cpu` for code that actually calls CUDA leaves GPUs hidden and
can fail at runtime; choosing `resource: gpu` for CPU work reserves a GPU that the
stage may not use.

Set `runner.slurm.partition_cpu` when CPU work must use a different partition.
Without it, CPU-routed stages request no GPUs on the runner's default partition.
A stage-specific partition remains available when one CPU stage needs different
routing:

```yaml
runner:
  kind: slurm
  slurm:
    partition:
      - gpu-general
      - gpu-overflow
    partition_cpu:
      - cpu-general
      - cpu-overflow

execution:
  stages:
    convert:
      strategy: single
      partition:
        - cpu-large-memory
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

`--dry-run` prints a submission-equivalent `sbatch` script for every planned
submission. Inspect the account, partition, nodes, tasks, GPU request, container,
mounts, working directory, and `srun` line before launching. The preview's
deterministic attempt ID, job name, and log path are replaced at launch. A
CPU-routed MIP script must omit both `#SBATCH --gpus-per-node` and
`srun --gpus-per-task` and must use the direct task launcher. A
validation-enabled MIP script must show the expected GPU count and a
`torchrun` launcher for the realization mesh.

Dry-run output and generated plan snapshots contain the configured setup hooks.
Never put literal credentials in `prerun_commands` or `postrun_commands`; the
loader rejects secret-like assignments. Inherit credentials from the scheduler
environment or source an access-controlled `setup_env` file.

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

The setup defaults keep `TMPDIR` at the short worker-local `/tmp` path and put
the vLLM, FlashInfer, Triton, and PyTorch kernel caches in explicit writable
directories there.
Preserve those commands for containerized workers: vLLM uses Unix-domain
sockets with a platform path limit, and a read-only container home prevents
the runtime caches from being initialized. Another short, worker-local writable
directory is also valid.
