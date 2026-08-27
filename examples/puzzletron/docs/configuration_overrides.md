# Configuration and experiment overrides

Puzzletron builds experiment settings from reusable YAML files in this
directory:

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

Choose where a built-in campaign stores its outputs without editing the YAML:

```bash
export PUZZLETRON_RUN_ROOT=/shared/puzzle_runs/my_campaign
```

Built-in experiment YAMLs use `PUZZLETRON_RUN_ROOT` as their `puzzle_dir`.
Generated bundles write the selected `puzzle_dir` directly. This directory
contains campaign outputs, manifests, resume information, and logs unless
`runner.slurm.log_dir` sends job logs elsewhere.

Run `orchestrate.py` with `--dry-run` after any configuration change. It
resolves and validates the experiment, runner, and execution files before job
submission, so misspelled or misplaced fields fail at the command boundary.

## Command-line overrides

Use command-line overrides for temporary experiment value changes. Append a
repeatable `--override KEY=VALUE` to the campaign command and inspect the
result with `--dry-run` before launch:

```bash
--override mip.runs.params-90.solver.num_solutions=4 \
--override ++runtime_annotations.reason=capacity-check \
--dry-run
```

Plain `KEY=VALUE` and explicit `++KEY=VALUE` both add or replace experiment
values. The `orchestrate.py` command and GPU jobs interpret these forms
identically.
Single-plus add (`+KEY=VALUE`) and delete (`~KEY`) operators are not
supported. Put structural changes in a copied run config so they remain easy to
review.

Overrides apply only to the experiment config. Edit or copy the runner and
execution files when changing site or scheduler settings.
