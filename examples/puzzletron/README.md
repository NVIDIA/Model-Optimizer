# Puzzletron v2

Puzzletron v2 helps you explore model shapes and select a smaller, faster
variant against your quality and deployment goals. Its guided setup creates a
reproducible, resumable campaign that compares candidates and can optionally
evaluate, benchmark, materialize, or distill the selected model.

## Table of contents

- [First campaign](#first-campaign)
- [Understand the campaign stages](#understand-the-campaign-stages)
- [Evaluate a checkpoint](#evaluate-a-checkpoint)
- [Configure a campaign](#configure-a-campaign)
- [Operate and recover a campaign](#operate-and-recover-a-campaign)
- [Extend Puzzletron](#extend-puzzletron)

## First campaign

The supported path is to prepare the control and worker environments, generate
a campaign, inspect and launch its smoke bundle, and then repeat the launch for
production. The same launch command resumes compatible work after an
interruption.

### 1. Prepare the environments

Create a lightweight environment for the setup wizard and controller:

```bash
python3 -m venv .venv-puzzletron-control
source .venv-puzzletron-control/bin/activate
python -m pip install \
  -r examples/puzzletron/requirements-setup.txt \
  -r examples/puzzletron/requirements-orchestrator.txt
```

GPU workers use the environment or container declared in the generated runner
file. Prepare the [worker environment](docs/environment_setup.md) before
launching. That guide covers supported containers, pinned CUDA and PyTorch
packages, patched dependencies, model-specific kernels, and verification.

### 2. Generate a campaign

Start the guided setup with the repository defaults:

```bash
python examples/puzzletron/puzzletron_setup_v2.py \
  --defaults examples/puzzletron/configs/setup/defaults.example.yaml
```

Choose **Balanced pruning** for a first campaign. For the maintained Qwen text
route, select `Qwen/Qwen3.5-0.8B` and the recommended Puzzle-KD v2 text dataset
or an existing worker-visible dataset. Review the detected model, worker and
scheduler settings, and output directory.

The wizard reads model configuration, not model weights, and does not submit
jobs. It writes validated `smoke/` and `production/` bundles plus a generated
`README.md`. Run any dataset preparation command in that generated README from
the worker environment before launch. See the
[setup wizard guide](docs/setup_wizard.md) for profiles, hosted datasets, full
configuration mode, generated files, and setup resume.

### 3. Inspect and launch smoke

Activate the control environment and inspect the generated smoke plan:

```bash
PUZZLETRON_BUNDLE=/path/to/generated/campaign/smoke

python examples/puzzletron/orchestrate.py \
  --experiment "$PUZZLETRON_BUNDLE/experiment.yaml" \
  --runner "$PUZZLETRON_BUNDLE/runner.yaml" \
  --execution "$PUZZLETRON_BUNDLE/execution.yaml" \
  --stage full --dry-run
```

Review the stage list, worker paths, execution strategies, resources, and
artifact root. `--stage full` runs every stage enabled by the generated
experiment in dependency order. Remove `--dry-run` to launch the smoke
campaign. The checked-in
[Qwen 3.5 0.8B smoke guide](docs/qwen3p5_0p8b_smoke.md) provides a separate
one-GPU MIP acceptance route when you need to validate that path directly.

### 4. Launch production

After smoke succeeds, change `smoke` to `production`, inspect that plan with
`--dry-run`, and launch it with the same command.

The experiment file owns model and algorithm choices, the runner file owns the
worker environment, and the execution file owns per-stage orchestration. Keep
all three together when launching or resuming a generated campaign.

### 5. Resume and inspect results

The controller shows live progress and stores durable state under
`${puzzle_dir}/orchestration/`. Run the same command with the same three files
to recover a detached or interrupted campaign; completed compatible stages are
not submitted again.

After the selected plan completes cleanly, the orchestrator attempts to write
the final report to
`<puzzle-dir>/artifacts/campaign_report/campaign_report.html`. A report failure
does not fail the completed campaign and is recorded in the controller result.
See [controller operations](docs/orchestration_operations.md) for individual
stages, `--once`, logging controls, security options, and recovery details, or
[campaign reports](docs/campaign_reports.md) to regenerate and interpret a
report.

## Understand the campaign stages

`--stage full` runs every stage enabled by the experiment in dependency order.
The generated dry-run plan is the exact stage and resource list for a campaign.
The complete pipeline is organized into these steps:

1. **Prepare inputs.** Convert the source checkpoint and, when configured,
   tokenize the campaign dataset.
2. **Measure pruning choices.** Collect width importance and optional depth
   importance or vLLM runtime statistics, then sort the teacher checkpoint.
3. **Validate and score.** Run the enabled sorting, width, slicing, and bypass
   sanity checks; collect bypass observations; build the replacement library;
   and score individual replacements.
4. **Search.** Solve the configured MIP runs to select candidate model shapes
   under parameter, memory, runtime, or quality constraints. See
   [MIP runs](docs/mip_profiles.md) for profiles, objectives, constraints,
   solution pools, and workload measurements.
5. **Process selected candidates.** Configured post-MIP flows can filter,
   evaluate, materialize, benchmark with AIPerf, and distill candidates. See
   [post-MIP pipelines](docs/post_mip_pipeline.md) for node types, branching,
   lineage, and downstream evaluation.
6. **Report.** The cumulative report records completed, pending, disabled, and
   optional work together with available results and warnings.

Sanity failures that show incorrect sorting or physical slicing block a valid
campaign result. Ranking-quality misses can remain visible as warnings. See
[sanity validation](docs/sanity_validation.md) for the checks, comparison
controls, and tolerances.

Run one stage with `--stage <stage_id>` only when its parent artifacts already
exist. It does not run missing prerequisites. Use `--stage full` for the normal
dependency-ordered campaign and whole-campaign resume.

## Evaluate a checkpoint

Candidate evaluation can be part of a post-MIP campaign flow, where metrics,
selection, materialization, and report lineage remain connected. Configure
that route with the
[post-MIP pipeline guide](docs/post_mip_pipeline.md#downstream-evaluation).

To evaluate a compatible local Hugging Face checkpoint without creating a
campaign, run the default one-GPU smoke in the Puzzletron worker environment:

```bash
python examples/puzzletron/evaluate_lmms_checkpoint.py \
  --checkpoint /path/to/checkpoint \
  --output-dir /path/to/results/checkpoint-smoke
```

The smoke evaluates eight samples each from IFEval and GSM8K. See
[checkpoint evaluation](docs/checkpoint_evaluation.md) for task selection,
full evaluation, model detection, runtime options, results, and troubleshooting.

## Configure a campaign

Generated bundles contain an experiment, runner, and execution file. The
default route above is sufficient when their generated values match your model
and infrastructure.

- Use [configuration and overrides](docs/configuration_overrides.md) to inspect
  the checked-in Hydra layout, choose artifact roots, or make temporary
  experiment changes.
- Use [Slurm configuration](docs/slurm_configuration.md) to change partitions,
  CPU-only stages, log locations, and accepted compatibility fields.
- Use the [setup wizard guide](docs/setup_wizard.md) to change profiles,
  datasets, generated files, or setup automation.

Run `--dry-run` after every configuration change. It resolves and validates
the experiment, runner, and execution files before any job is submitted.

## Operate and recover a campaign

See [controller operations](docs/orchestration_operations.md) for individual
stages, `--once`, non-interactive behavior, logging controls, security options,
execution strategies, controller records, and recovery. Remote model code and
online tokenizer resolution remain disabled by default and should be enabled
only for trusted sources.

To run with an agent, ask it to use
[`running-puzzletron`](../../.agents/skills/running-puzzletron/SKILL.md) and
provide the model, dataset, compute environment, search space, resource
constraints, and required downstream stages.

## Extend Puzzletron

- [Architecture](docs/v2_architecture.md) describes the stage registry,
  campaign DAG, scheduler-neutral control plane, and maintainer guidance.
- [Legacy Nano campaign](docs/legacy_nano_campaign.md) describes the separate
  online evaluation and finalist-materialization workflow used by the
  checked-in Nano configuration.
