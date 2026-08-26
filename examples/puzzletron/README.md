# Puzzletron v2

Puzzletron v2 helps you explore model shapes and select a smaller, faster
variant against your quality and deployment goals. Its guided setup creates a
reproducible, resumable campaign that compares candidates and can optionally
distill the selected model.

## Run a campaign

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
launching; that guide covers the supported container, pinned CUDA and PyTorch
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
the worker environment before launch.

### 3. Inspect and launch

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
experiment in dependency order, including evaluation, materialization,
benchmarking, or distillation when selected during setup. Remove `--dry-run`
to launch the smoke campaign. After it succeeds, change `smoke` to
`production`, inspect that plan, and launch it with the same command.

The experiment file owns model and algorithm choices, the runner file owns the
worker environment, and the execution file owns per-stage orchestration. Keep
all three together when launching or resuming a generated campaign.

### 4. Resume and inspect results

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

## Other workflows

- **Evaluate a checkpoint without a campaign:** run
  `python examples/puzzletron/evaluate_lmms_checkpoint.py --checkpoint
  /path/to/checkpoint --output-dir /path/to/results/checkpoint-smoke`. See
  [checkpoint evaluation](docs/checkpoint_evaluation.md) for task selection and
  advanced options.
- **Run the checked-in one-GPU MIP acceptance route:** follow the
  [Qwen 3.5 0.8B smoke guide](docs/qwen3p5_0p8b_smoke.md). This is separate from
  the generated end-to-end campaign above.
- **Run with an agent:** ask an agent to use
  [`running-puzzletron`](../../.agents/skills/running-puzzletron/SKILL.md) and
  provide the model, dataset, compute environment, search space, resource
  constraints, and required downstream stages.

## Customize and operate

The default route above is sufficient for a generated campaign. Use these
focused guides only when you need to change or inspect a specific part:

- [Setup wizard](docs/setup_wizard.md): profiles, full configuration mode,
  hosted datasets, generated files, and setup resume.
- [Configuration and overrides](docs/configuration_overrides.md): checked-in
  config layout, run roots, and temporary campaign changes.
- [Slurm configuration](docs/slurm_configuration.md): partitions, CPU-only
  stages, logs, and compatibility fields.
- [MIP runs](docs/mip_profiles.md): objectives, constraints, solution pools,
  workload measurements, and search variants.
- [Post-MIP pipelines](docs/post_mip_pipeline.md): evaluation, filtering,
  materialization, AIPerf, and distillation.
- [Sanity validation](docs/sanity_validation.md): correctness checks,
  tolerances, ranking warnings, and qualification guidance.
- [Legacy Nano campaign](docs/legacy_nano_campaign.md): the separate online
  evaluation and finalist-materialization workflow.
- [Architecture](docs/v2_architecture.md): stage registry, campaign DAG,
  control plane, and maintainer guidance.
