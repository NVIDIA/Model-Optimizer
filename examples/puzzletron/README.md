# Puzzletron

Puzzletron runs resumable pruning campaigns. A maintained run has two authored
inputs:

- a small recipe that selects the model, workflow, mode, run directory, and
  resource profile;
- one reusable site file for worker paths, environment, scheduler, and
  available capacity.

Puzzletron resolves the full stage configuration, validates it, and stores an
immutable copy with the run.

- [Start here: lifecycle smoke](#start-here-lifecycle-smoke)
- [Choose the next task](#choose-the-next-task)
- [Results and live progress](docs/campaign_reports.md)
- [Central results catalog](reports/catalog.yaml)
- [Documentation map](#documentation-map)

## Quickstart

Use Python 3.10 through 3.14 for the lightweight controller environment:

```bash
python3 -m venv .venv-puzzletron
source .venv-puzzletron/bin/activate
python -m pip install --upgrade pip
python -m pip install -r examples/puzzletron/requirements-setup.txt
```

Create one site file and fill in its worker-visible checkout, virtual
environment, shared Hugging Face cache, Slurm account, and partition:

```bash
cp examples/puzzletron/configs/site.example.yaml puzzletron.site.yaml
```

Choose a checked-in recipe. Start with the Qwen 3.5 0.8B VLM smoke, which
checks the full image-text lifecycle on two colocated GPUs:

```bash
cp examples/puzzletron/configs/recipes/qwen3p5_0p8b_vlm_smoke.yaml puzzletron.recipe.yaml
# Edit puzzletron.recipe.yaml and set run_root to a new shared path.
```

Validate the inputs, inspect their provenance and resources, and preview the
scheduler commands:

```bash
python examples/puzzletron/puzzletron.py validate puzzletron.recipe.yaml \
  --site puzzletron.site.yaml
python examples/puzzletron/puzzletron.py explain puzzletron.recipe.yaml \
  --site puzzletron.site.yaml
python examples/puzzletron/puzzletron.py dry-run puzzletron.recipe.yaml \
  --site puzzletron.site.yaml
```

Launch the same pair:

```bash
python examples/puzzletron/puzzletron.py launch puzzletron.recipe.yaml \
  --site puzzletron.site.yaml
```

Resume and inspect by run directory. Resume uses the sealed inputs stored with
the run, not the current recipe or site file.

```bash
python examples/puzzletron/puzzletron.py resume /shared/puzzle_runs/my-run
python examples/puzzletron/puzzletron.py inspect /shared/puzzle_runs/my-run
python examples/puzzletron/puzzletron.py results inspect /shared/puzzle_runs/my-run
```

Use a new run directory whenever an authored input changes. Completed
compatible stages are not rerun, and `resume` always uses the sealed inputs
under the existing run directory. See [run and
recovery](docs/orchestration_operations.md) for allocation replacement and
retry behavior.

Puzzletron stores scheduler state and resolved bundles under
`<run-root>/orchestration/`. The authoritative run, stage, progress, subject,
metric, artifact, provenance, and timing evidence is atomically refreshed in
`<run-root>/results/result.json`.

## Choose a recipe

Run `python examples/puzzletron/puzzletron.py routes` to list the maintained
model, workflow, and mode combinations. Five checked-in recipes cover Qwen 3.5
0.8B text and VLM smokes, a larger 0.8B VLM campaign, and 4B VLM smoke and
campaign routes. See [maintained recipes](docs/maintained_recipes.md) for their
requirements and interpretation limits.

## Read results and progress

After the selected plan completes cleanly, Puzzletron finalizes `result.json`,
then generates the optional HTML view at
`<run-root>/artifacts/campaign_report/campaign_report.html`. The HTML contains
no unique evidence and can be regenerated from the result. See the
[central results catalog](reports/catalog.yaml) for retained runs and
[campaign reports](docs/campaign_reports.md) for detached inspection, export,
refresh, evidence drill-down, and interpretation. See
[run and recovery options](docs/orchestration_operations.md) for individual
stages, `--once`, logging controls, security options, and recovery details. For
a failed or interrupted run, follow the actionable checks in
[run and recovery options](docs/orchestration_operations.md#progress-and-interruption).

Recipes are the public run interface. Files under `configs/families/` are
internal composition templates and should not be edited or launched directly.

## Custom models

If no maintained recipe matches the model, use the existing setup wizard
through the same main command:

```bash
python examples/puzzletron/puzzletron.py setup
```

The wizard creates smoke and production bundles and a README with their launch
commands. It does not submit jobs. Existing `puzzletron_setup.py`,
`puzzletron_setup_v2.py`, and three-file configurations remain supported for
compatibility, but they are not additional maintained-recipe workflows.

## Documentation

- [Maintained recipes](docs/maintained_recipes.md): choose and run the five
  supported recipe routes.
- [Environment setup](docs/environment_setup.md): configure controller and
  worker environments.
- [Configuration](docs/configuration.md): recipe and site fields,
  advanced changes, provenance, and sealed bundles.
- [Run and recovery](docs/orchestration_operations.md): progress, retries,
  interruption, inspection, and resume.
- [Evaluation](docs/post_mip_pipeline.md): checkpoint materialization,
  evaluation, serving, and distillation stages.
- [Campaign reports](docs/campaign_reports.md): generate and interpret the
  cumulative report.
- [Central results catalog](reports/catalog.yaml): discover retained structured
  results and qualified historical reports.

Additional evaluator and implementation references live under
`examples/puzzletron/docs/`.
