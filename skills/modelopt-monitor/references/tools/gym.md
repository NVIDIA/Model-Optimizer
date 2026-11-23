# NeMo Gym evaluation jobs (`type: gym`)

- **Register** the Gym run id (the invocation id the Gym skill prints, `gym-job-<timestamp>-<hash>`)
  as one `type: gym` entry, and each Slurm job id of its chain as a `type: slurm` entry so the
  terminal check below applies.
- **Run-level state** (health, coverage, recovery decisions) is owned by the Gym evaluation
  skill (`run-and-babysit-eval`: its inspect and recovery steps). This skill does not re-derive
  it; it keeps the registry and reports the Gym skill's verdict when it changes.
- **Do not** poll a Gym run with `nel` commands, and do not cancel jobs of a run that is not yours.
