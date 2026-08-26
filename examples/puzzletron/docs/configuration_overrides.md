# Configuration and experiment overrides

Checked-in experiment configs use Hydra composition:

```text
examples/puzzletron/configs/
├── base.yaml                         # pipeline-wide defaults
└── families/
    └── <family>/
        ├── family.yaml               # descriptors, hooks, and family axes
        └── <model>/
            ├── model.yaml            # checkpoint metadata and legal domains
            └── runs/<run>.yaml       # exact named campaign run
```

Set a site-specific artifact root without editing a checked-in config:

```bash
export PUZZLETRON_RUN_ROOT=/shared/puzzle_runs/my_campaign
```

Checked-in experiment YAMLs use `PUZZLETRON_RUN_ROOT` to resolve `puzzle_dir`.
Generated bundles write their selected `puzzle_dir` directly. In both cases,
`puzzle_dir` is the canonical location for artifacts, manifests, controller
state, and logs unless `runner.slurm.log_dir` relocates attempt logs.

## Command-line overrides

Use command-line overrides for temporary experiment value changes. Append a
repeatable `--override KEY=VALUE` to the orchestrator command and inspect the
result with `--dry-run` before launch:

```bash
--override mip.runs.params-90.solver.num_solutions=4 \
--override ++runtime_annotations.reason=capacity-check \
--dry-run
```

Plain `KEY=VALUE` and explicit `++KEY=VALUE` both add or replace experiment
values. The controller and GPU workers interpret these forms identically.
Single-plus add (`+KEY=VALUE`) and delete (`~KEY`) operators are not
supported. Put structural changes in a copied run config so they remain easy to
review.

Overrides apply only to the experiment config. Edit or copy the runner and
execution files when changing site or scheduler settings.
