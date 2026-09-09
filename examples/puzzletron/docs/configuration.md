# Configuration

A maintained run has two user-owned files:

- The recipe selects a maintained model, workflow, mode, run directory, and
  named site resource. It may also provide required data identity and explicit
  advanced changes.
- The site file owns the worker environment, shared cache, scheduler or
  bare-metal hosts, and named resource capacities. Reuse it across runs.

Run `puzzletron.py routes` to list supported combinations. Files under
`configs/families/` and the internal route catalog supply model facts and stage
defaults. They are implementation inputs, not additional public recipes.

## Resolution

Puzzletron validates the recipe and site, selects the internal facts for the
route, applies explicit recipe changes, selects the named site resource, and
compiles the full stage plan. Both public schemas are closed. Validation
rejects duplicate YAML keys, unknown or unused fields, unsupported routes,
no-op changes, settings for inactive stages, invalid meshes, and plans beyond
site capacity.

Use `explain` to inspect the winning source and allocation for every value:

```bash
python examples/puzzletron/puzzletron.py explain puzzletron.recipe.yaml \
  --site puzzletron.site.yaml
```

Relative `run_root` values resolve from the controller working directory.
Relative `data.path` values resolve inside that run directory. Use absolute
worker-visible paths for shared data stored elsewhere.

## Advanced changes

Normal recipes omit implementation settings. Use `advanced.experiment` only
to replace an existing internal value:

```yaml
advanced:
  experiment:
    mip.runs.params-90.solver.num_solutions: 4
    pruning.eval_samples: 32
```

Execution changes use the runtime compiler's stage schema:

```yaml
advanced:
  execution:
    stages:
      post.candidate-evaluation.screening_kd:
        parallel:
          tp: 8
          pp: 4
          cp: 1
          dp_shard: 1
          dp_replicate: 2
          ep: 1
```

The model-instance GPU count is `TP * PP * CP * DP_SHARD * DP_REPLICATE`.
Expert parallelism overlays the sharded data-parallel dimension. Independent
`instances` multiply the total request. Site-owned partitions and capacities
cannot be changed from a recipe. Accepted changes and their sources are saved
in `provenance.json`.

## Sealed run bundle

`dry-run` and `launch` write a content-addressed bundle under
`<run-root>/orchestration/resolved_bundles/`. It contains the normalized
recipe, runtime and audit experiment snapshots, runner and execution contracts,
compiled plan, provenance, source identity, and file hashes.

Generated YAML files begin with a `DO NOT EDIT` header.
`experiment.runtime.yaml` is the executable worker input;
`experiment.resolved.yaml` is an audit view. Launch binds a run directory to
one bundle. Resume verifies and reuses that bundle instead of reading current
recipes, site settings, or internal defaults. Change an authored input only for
a new run directory.

## Custom models and existing configurations

When no maintained recipe fits, run the existing custom-model wizard through:

```bash
python examples/puzzletron/puzzletron.py setup
```

It creates self-contained smoke and production bundles and does not submit
jobs. Their generated README contains the launch commands. Existing
`puzzletron_setup.py`, `puzzletron_setup_v2.py`, and experiment/runner/execution
configurations remain supported for compatibility. They are not alternative
authoring paths for a new maintained-route run.

The large-model topology tests validate schema and capacity calculations only;
they are not runtime validation for an unlisted model. Generated bundles live
under ignored run directories and should not be committed.
