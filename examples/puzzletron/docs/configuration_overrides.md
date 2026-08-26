# Experiment overrides

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
