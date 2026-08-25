# Controller operation and recovery

Run one stage with the same experiment, runner, and execution files used for a
full campaign:

```bash
PUZZLETRON_BUNDLE=/path/to/generated/campaign/production

python examples/puzzletron/orchestrate.py \
  --experiment "$PUZZLETRON_BUNDLE/experiment.yaml" \
  --runner "$PUZZLETRON_BUNDLE/runner.yaml" \
  --execution "$PUZZLETRON_BUNDLE/execution.yaml" \
  --stage width_importance
```

The launch command runs a foreground controller. It submits every
dependency-ready branch concurrently, polls scheduler state, and exits when the
selected plan completes or fails.

## Progress and interruption

Interactive terminals show a live stage table with status, resources, elapsed
time, and a best-effort ETA when a stage reports progress. Completed stages,
dependency waits, failures, and descendants blocked by failures remain visible.
Redirected output uses timestamped one-line updates instead.

Press `q` or Ctrl-C in an interactive terminal to cancel active jobs and quit,
detach while leaving jobs running, or continue. Non-interactive Ctrl-C and
SIGTERM cancel active work and quit. A detached controller preserves durable
handles, so running the same command recovers the active jobs.

Use `--color always` when piping through `tee`, `--color never` for plain logs,
and `--poll-interval SECONDS` to change the default five-second poll interval.

## State and execution records

Durable controller state is written under `${puzzle_dir}/orchestration/`. The
controller supports `single`, `sharded`, and `persistent_pool` strategies,
Slurm and SSH executors, attempt recovery, and semantic stage validation. See
the [`configs/orchestration/`](../configs/orchestration/) directory for starter
runner and execution files.

Accepted rank-zero stage results also write checksum-validated execution
records under `<puzzle-dir>/manifests/executions/`. Puzzletron validates these
records when resuming a stage. They identify existing outputs but do not copy
or make those outputs immutable.
