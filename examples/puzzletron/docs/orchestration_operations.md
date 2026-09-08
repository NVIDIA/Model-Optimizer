# Run and recover campaigns

The public `launch` and `resume` commands execute the complete sealed dependency
plan. This keeps the recorded plan identical to the launched plan.

Use the public `dry-run` subcommand after changing the recipe or site file or
updating the checkout. For Slurm sites, each dry-run submission includes a
submission-equivalent `sbatch` script with its nested `srun` command. CPU/GPU
requests, task launchers, partitions, containers, mounts, and worker commands
match a launch. The preview uses a deterministic attempt ID, job name, and log
path; a real launch replaces those three identifiers.

When the selected resource profile uses `reusable_allocation`, dry-run instead shows
one outer Slurm script plus the logical attempts that share it. The public
launch command is unchanged. Puzzletron persists the outer handle, reattaches
when the same command finds it active, and runs the ordinary controller inside
the allocation. A completed campaign with compatible artifacts and a matching
terminal allocation result submits no new allocation.

In per-attempt mode, the launch command runs the controller in the foreground.
It submits every dependency-ready branch concurrently, polls scheduler state,
and exits when the selected plan completes or fails. In reusable mode, the
foreground process watches the outer allocation; the inner controller writes
its logical-stage progress to the allocation log whose path the watcher prints.

`launch --once` recovers and polls existing attempts, submits currently ready work,
and exits after one scheduling iteration. Submitted jobs keep running. Invoke
the same `--once` command again for the next recovery and scheduling iteration.
In reusable mode, `--once` submits or reattaches to the one outer allocation and
then exits; the controller inside that allocation continues the campaign.

Stage resources are defaults, not fixed deployment policy. In per-attempt Slurm
mode they select zero-GPU or GPU scheduler requests. In reusable mode they
select a GPU lease or zero-GPU subprocess inside the outer GPU allocation;
stage-specific CPU partitions and CPU scheduler sizing do not apply there.
Local and SSH bare-metal execution use the same plan and control which GPUs are
visible to the worker. An explicit `execution.stages.<stage>.resource` value
overrides the default, including on an interactive GPU node. A CPU override can
fail if the stage actually calls CUDA; a GPU override for CPU work can reserve
an unused GPU.

Remote model code and AIPerf v0.11 online tokenizer resolution are disabled by
default. Enable remote code only for a trusted model source. The tokenizer
compatibility option permits the AIPerf child process to resolve its tokenizer
online even when the surrounding campaign is configured for offline loading.

## Progress and interruption

The controller shows a live stage table in an interactive terminal with status,
resources, elapsed time, the active log path, and a best-effort ETA after it
measures item throughput. Completed stages, dependency waits, failures, and
descendants blocked by failures remain visible. Redirected controller output
emits a heartbeat every 30 seconds with completed/total stages, queued/running
jobs, elapsed time, each active stage's state and progress, its log path, and a
measured ETA. It says `ETA unavailable` until it has enough evidence. In
reusable mode this controller output is inside the allocation log; the outer
watcher itself reports allocation state transitions and that log path.

Press `q` or Ctrl-C in an interactive terminal to cancel active jobs and quit,
detach while leaving jobs running, or continue. Non-interactive Ctrl-C and
SIGTERM cancel active work and quit. Detaching preserves saved job information,
so running the same command recovers the active jobs.

The reusable-allocation watcher has a narrower interruption contract: Ctrl-C detaches and leaves the outer Slurm job running. Rerun the identical command to reattach. Cancel the outer job with the site's normal Slurm command only when you intend to stop all work in that allocation. After Slurm reports the outer job cancelled or failed, rerunning starts a replacement unless a compatible worker result records clean completion or an uncancelled terminal stage failure. Completed logical stages remain skipped, and in-flight local attempts from that allocation are recorded as cancelled before unfinished work is resubmitted. A recorded cancellation and a final-report failure remain retryable. Use a new run root before rerunning a public recipe with a terminal stage failure. A legacy three-file run can instead change its configuration before rerunning.

For a public recipe run, do not edit the generated experiment, runner, or
execution files. Change the recipe or site file and use a new run root; the
existing run root remains bound to its sealed bundle and is resumed with
`puzzletron.py resume`.

Redirect stderr before piping through `tee` (for example, append
`2>&1 | tee run.log`) so progress output is captured. Use `--color always` for
colored output, `--color never` for plain logs, and `--poll-interval SECONDS`
to change the default five-second poll interval.

## State and execution records

The recipe calls the campaign output directory `run_root`. Resume information
and the sealed resolved bundle are written under `<run-root>/orchestration/`.
`resume <run-root>` verifies and reuses the sealed experiment, runner,
execution, plan, manifest, and provenance records. The runtime
supports `single`, `sharded`, and `persistent_pool` strategies, Slurm and SSH
executors, attempt recovery, and semantic stage validation.

Accepted rank-zero stage results also write checksum-validated execution
records under `<run-root>/manifests/executions/`. Puzzletron validates these
records when resuming a stage. They identify existing outputs but do not copy
or make those outputs immutable. The configuration bundle is separately
immutable and hash-checked.

Dataset preparation additionally records the expected files. Routine resume
checks their paths, types, sizes, and timestamps without rereading every image
or video. If metadata changed, Puzzletron verifies that file's SHA-256; set
`prepare_dataset.verify_content: true` to hash every file on each resume.
This is a preventive corruption check, not a response to a known failed
campaign.

A different engineer can prepare the same dataset into another local path. Pin
`prepare_dataset.revision` or `data.revision` to an immutable source revision to
obtain the same acquisition, then run the preparation stage to create a local
manifest. Existing unowned or mismatched output directories are preserved and
must be moved aside or rebuilt rather than overwritten.
