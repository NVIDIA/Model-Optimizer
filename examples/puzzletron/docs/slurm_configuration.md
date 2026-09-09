# Slurm configuration

Use one site file for worker and Slurm settings. Its named resource profiles
provide GPU capacity and scheduling mode. Maintained route defaults own normal
stage behavior; rare stage changes belong under the recipe's explicit
`advanced.execution` section.

Every Slurm site must set `site.slurm.account`. Container mounts are accepted
only with a container, and bare-metal sites reject container settings because
their executor does not consume them.

## Reusable single-node allocations

The resource profile mode `per_attempt` submits one Slurm job for each logical
attempt. Select `mode: reusable_allocation` to submit one outer
Slurm job and run the existing dependency-aware controller inside it. This is
additive: stage records, attempts, logs, artifacts, failure policies, and
resume validation remain separate even though they share one container and
allocation.

Reusable mode currently supports one node. Compilation rejects any stage whose
resolved topology requires more than one. The selected profile's `gpus_per_node`
sets the outer allocation capacity; logical attempts lease disjoint subsets of
the GPUs visible inside that allocation. Stage `instances` and task topology
control how much of that capacity each ready stage can use. Capacity is not
hard-coded to eight. The maintained Qwen 3.5 0.8B smoke reserves two GPUs on
one node, while the representative campaign reserves one eight-GPU node. Both
profiles have been qualified only in the documented eight-GPU node
environment.

This mode assumes a Slurm site where one containerized task can see the full
node allocation, shared campaign paths are mounted identically, and runtime
caches are writable for the lifetime of the outer job. The outer job uses the
site's CPU and memory defaults for its GPU request, which must be sufficient for
the concurrent workers. Other sites may need to adapt the site's account,
partition, container integration, mounts, time limit, GPU capacity, and cache
hooks. Use per-attempt mode when those assumptions do not hold.

### Stage instances

`execution.stages.<stage>.instances` is the maximum number of independent workers that a stage may use. It controls execution concurrency, not how many student candidates the experiment creates. Candidate counts come from the MIP and post-MIP flow, including `num_solutions` and filtering rules such as `top_k`.

The meaning depends on the stage strategy:

- `single` requires one instance.
- `sharded` uses up to that many workers for independent shards. Candidate `evaluation` and `downstream_evaluation` stages may use fewer workers when fewer candidates are available.
- `persistent_pool` starts that many resident workers. Replacement scoring distributes its workers across configured hidden-width scenarios, with at least one worker per scenario.

Puzzletron does not generally derive `instances` from MIP `num_solutions` or post-MIP `top_k`. Set both values intentionally when a filter leaves fewer candidates than a downstream GPU stage's configured worker count.

Each instance consumes the stage's resolved GPUs per instance. In reusable mode,
compilation rejects a stage that cannot fit within the selected resource
profile's `gpus_per_node`; the controller starts ready stages only when enough
of that capacity is free. For example, five one-GPU candidate instances use at
most five GPUs concurrently. Lower `instances` to reduce concurrency or
accommodate a larger per-instance parallel mesh.

### Interruption and resume

If the launching terminal exits while the outer Slurm job is still active, the job continues. Running the same command reattaches to its recorded job instead of submitting a duplicate.

If the allocated node fails or the outer job reaches its Slurm time limit, all subprocesses inside that allocation stop. Run the same command again after Slurm reports the job as terminal. Unless the compatible worker result already records clean completion or an uncancelled terminal stage failure, Puzzletron submits a replacement allocation, skips stages with complete validated artifacts, and retries incomplete work. A recorded cancellation and a final report failure remain retryable. An incomplete stage resumes from its own checkpoints only when that stage supports native resume; otherwise its unfinished attempt runs again.

The generated job uses Slurm's no-requeue behavior, so replacement requires a
new Puzzletron invocation or external automation. `site.slurm.time_limit`
covers the complete reusable campaign, not each stage separately. Allow enough
headroom for the full campaign, or use per-attempt mode when independent
scheduler failure domains are more important than avoiding repeated startup
overhead.

## Partitions and logs

`site.slurm.partition` sets the default allocation partition. It accepts one
partition name or a list of eligible names. Omit it to use the site's Slurm
default. A selected resource profile may replace this partition for the whole
concise-recipe plan. Stage-specific partitions belong to external or
wizard-generated execution files; `advanced.execution` cannot override
site-owned partition fields. In reusable mode, `site.slurm.partition_cpu` does
not apply to zero-GPU subprocesses inside the outer allocation.

`site.slurm.log_dir` sets the directory used for every attempt log, including
the final-report attempt. When omitted, logs are written below
`<run-root>/logs`. Relative values are resolved from the run root; absolute
paths are used as written.

The site loader accepts `partition_interactive`, `partition_batch`, and
`interactive_max_nodes` as compatibility fields. They infer stage routing from
role names and node count, which assumes a particular site layout. Maintained
concise-route sites use `site.slurm.partition`, the supported `partition_cpu`
fallback, and resource-profile partitions instead.

The production examples also avoid literal `interactive` and `batch` stage
overrides because those partition names are not portable between Slurm sites.

## Stage resource defaults

Puzzletron defaults data preparation, checkpoint conversion, library building,
MIP solving, filtering, and reporting to `resource: cpu`. Model execution stages
default to `resource: gpu`. In per-attempt mode these values control each
scheduler request. In reusable mode they select either a GPU lease or a
zero-GPU subprocess inside the already-running allocation.

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

In per-attempt mode, set `site.slurm.partition_cpu` when CPU work must use a
different partition. Without it, CPU-routed stages request no GPUs on the
site's default partition:

```yaml
site:
  kind: slurm
  slurm:
    partition:
      - gpu-general
      - gpu-overflow
    partition_cpu:
      - cpu-general
      - cpu-overflow
```

Slurm selects one partition from each eligible list. The
[`site.example.yaml`](../configs/site.example.yaml) shows the concise public
site and resource-profile contract. Wizard-generated and external execution
files retain their existing stage-partition controls.

For per-attempt CPU stages, `site.slurm.cpu_cpus_per_task` sets the requested
CPU count and `site.slurm.cpu_memory_mb` sets memory in MiB. Omit them to use
the site's Slurm defaults. These settings are part of resume identity, so
changing either causes Puzzletron to submit the stage with the new allocation
instead of treating an older active attempt as the same work. They do not
change the reusable outer GPU job.

In per-attempt mode, `--dry-run` prints a submission-equivalent `sbatch` script
for every planned submission. In reusable mode, it prints one outer `sbatch`
script plus every logical attempt and resource request; logical attempts have
no separate scheduler script. Inspect the account, partition, nodes, tasks, GPU
request, container, mounts, working directory, and `srun` line before
launching. The preview's deterministic attempt ID, job name, and log path are
replaced at launch. A CPU-routed MIP script in per-attempt mode must omit both
`#SBATCH --gpus-per-node` and `srun --gpus-per-task` and must use the direct task
launcher. A validation-enabled MIP script must show the expected GPU count and
a `torchrun` launcher for the realization mesh.

Dry-run output and generated plan snapshots contain the configured setup hooks.
Never put literal credentials in `prerun_commands` or `postrun_commands`; the
loader rejects secret-like assignments. Inherit credentials from the scheduler
environment or source an access-controlled `setup_env` file.

## Scheduler settings and model settings

Do not put `sequence_parallel` under
`advanced.execution.stages.<stage>.parallel`. That mapping controls scheduler allocation
and accepts mesh dimensions such as `tp`, `pp`, and `dp_replicate`.
`sequence_parallel` changes model execution and belongs in the maintained
model or workflow profile, not in scheduler allocation.

Recipe and site files reject unknown fields and suggest the closest valid name
when possible.

## Worker setup hooks

`site.environment.prerun_commands` and `postrun_commands` are copied
into generated worker scripts and appear in dry-run output. Puzzletron rejects
obvious literal assignments to credential-like variables so those values are
not persisted. Inherit credentials from the launch environment, require an
existing variable such as `${API_KEY:?set API_KEY}`, retrieve it from a secret
command, or source a permission-protected `setup_env` file. This check catches
common mistakes but is not a shell parser or a complete credential scanner.

For containerized workers, use `prerun_commands` when the site needs to place
`TMPDIR` or runtime caches in a short, worker-local writable directory. vLLM
uses Unix-domain sockets with a platform path limit, and a read-only container
home prevents runtime caches from being initialized.
