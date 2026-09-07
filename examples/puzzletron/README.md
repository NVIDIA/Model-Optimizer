# Puzzletron v2

Puzzletron v2 helps you explore model architectures and find smaller, faster
variants that meet your quality and deployment goals. Guided setup creates
reproducible campaigns that find and compare candidates. Campaigns can
evaluate, benchmark, materialize, or distill those candidates.

## Table of contents

- [Start here: lifecycle smoke](#start-here-lifecycle-smoke)
- [Understand the campaign stages](#understand-the-campaign-stages)
- [Evaluate a checkpoint](#evaluate-a-checkpoint)
- [Configure a campaign](#configure-a-campaign)
- [Operate and recover a campaign](#operate-and-recover-a-campaign)
- [Extend Puzzletron](#extend-puzzletron)

## Start here: lifecycle smoke

Start with the maintained Qwen 3.5 0.8B VLM smoke. It exercises the complete
image-text lifecycle with bounded workloads and at most one GPU per stage. Once
it succeeds, the optional longer example campaign uses the same environment,
runner, orchestrator, progress display, report, and resume command. That larger
campaign is a scheduled, multi-hour example, not a routine smoke or presubmit.

### 1. Create the controller environment

From the ModelOpt checkout, use Python 3.10 through 3.14 to create a lightweight
virtual environment for the command that plans, launches, and resumes the
smoke:

```bash
python3 --version  # must report Python 3.10 through 3.14
python3 -m venv .venv-puzzletron
source .venv-puzzletron/bin/activate
python -m pip install --upgrade pip
python -m pip install -r examples/puzzletron/requirements-setup.txt
```

This controller environment does not need PyTorch or model weights. Model
conversion, evaluation, serving, and distillation run in the worker environment
selected by the runner.

### 2. Configure the worker runner

Use a reviewed Puzzletron worker image supplied by your site. Create one runner
file from the template if your site does not already provide one:

```bash
cp examples/puzzletron/configs/orchestration/runner.slurm.example.yaml \
  runner.slurm.yaml
```

Set the Slurm account and partitions, worker image, and any shared-storage
mounts in `runner.slurm.yaml`. Cache, dataset, and run-root paths must be
worker-visible at their configured locations. The runner's repository and
Python paths must exist inside the worker image. See [environment
setup](docs/environment_setup.md) for the worker contract and [Slurm
configuration](docs/slurm_configuration.md) for each runner field.

The three campaign inputs have separate jobs:

- The experiment file defines the model, data, and bounded smoke workload. Do
  not edit it for this first run.
- The runner file contains the site-specific scheduler and worker settings.
- The execution file maps the smoke stages onto one GPU or a CPU worker.

### 3. Select shared paths

Set a Hugging Face cache visible to workers and a new run root on shared
storage:

```bash
export HF_HOME=/path/to/shared/huggingface-cache
export PUZZLETRON_SOURCE_REVISION="$(git rev-parse HEAD)"
export PUZZLETRON_DATASET_REVISION=51f4f4d219315c3283950994d4eb3d7fc30aa87b
export PUZZLETRON_RUN_ROOT=/path/to/qwen3p5_0p8b_vlm_smoke
export PUZZLETRON_RUN_ROOT="$(python -c 'import os; print(os.path.realpath(os.environ["PUZZLETRON_RUN_ROOT"]))')"
```

The smoke's first stage downloads, validates, and caches its eight Nemotron-VLM
image-conversation samples and the pinned evaluation rows. No PyTorch data
command or separate cache script runs in the controller venv. Offline workers
must use a cache prepared as described in the
[VLM evaluation guide](docs/vlm_checkpoint_evaluation.md#cache-benchmark-data).

### 4. Inspect and run the smoke

Activate `.venv-puzzletron`, select the maintained files, and inspect the exact
plan before it requests resources:

```bash
EXPERIMENT=examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/full_vlm_smoke.yaml
EXECUTION=examples/puzzletron/configs/orchestration/execution.single_gpu.yaml
RUNNER=runner.slurm.yaml

python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full --dry-run
```

Review the resolved paths, CPU and GPU requests, worker image, mounts, and log
locations. Then run the same plan without `--dry-run`:

```bash
python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full
```

### 5. Resume and inspect results

Run the launch command above again to recover an interrupted smoke or verify a
completed one. Compatible completed stages are not submitted again. Puzzletron
stores its structured runtime state under
`$PUZZLETRON_RUN_ROOT/orchestration/`.

While the command is running, it shows completed/total stages, queued and
running stages, elapsed time, 30-second heartbeats, and the active log path.
Stages expose their native inner unit when available, including evaluation
samples and measured rate. An ETA appears only after progress provides a
reliable denominator and the controller observes throughput; otherwise it says
`ETA unavailable`.

After the selected plan completes cleanly, `orchestrate.py` attempts to write the
final report to
`<puzzle-dir>/artifacts/campaign_report/campaign_report.html`. A report failure
does not fail the completed campaign and is recorded in the run result. See
[run and recovery options](docs/orchestration_operations.md) for individual
stages, `--once`, logging controls, security options, and recovery details, or
[campaign reports](docs/campaign_reports.md) to regenerate and interpret a
report. For a failed or interrupted run, follow the actionable checks in
[run and recovery options](docs/orchestration_operations.md#progress-and-interruption).

The [Qwen VLM example guide](docs/qwen3p5_0p8b_vlm_smoke.md) lists the smoke's
expected lifecycle checks and explains how to interpret its bounded results.

### 6. Optional: run the longer multi-hour example

After the smoke succeeds, keep the controller venv and runner and select a new
run root plus the longer example files:

```bash
EXPERIMENT=examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/vlm_campaign.yaml
EXECUTION=examples/puzzletron/configs/orchestration/qwen3p5_0p8b/execution.vlm_campaign.yaml
export PUZZLETRON_RUN_ROOT=/path/to/qwen3p5_0p8b_vlm_campaign
export PUZZLETRON_RUN_ROOT="$(python -c 'import os; print(os.path.realpath(os.environ["PUZZLETRON_RUN_ROOT"]))')"

python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full --dry-run
```

Inspect the larger plan, then remove `--dry-run` to launch it. Use that same
launch command for every resume. The example increases sample counts,
candidate coverage, and distillation work, but does not introduce another
operational path. Use focused tests and the lifecycle smoke for routine
development; reserve this campaign for scheduled integration validation.

## Understand the campaign stages

`--stage full` runs every stage enabled by the experiment in dependency order.
The generated dry-run plan is the exact stage and resource list for a campaign.
The complete pipeline is organized into these steps:

1. **Prepare inputs.** Convert the source checkpoint and, when configured,
   tokenize the campaign dataset.
2. **Measure pruning choices.** Collect importance for hidden width (the
   model's transformer hidden size) and optional depth importance or vLLM
   runtime statistics, then sort the teacher checkpoint.
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
[post-MIP pipeline guide](docs/post_mip_pipeline.md#evaluate-saved-checkpoints).

For standalone evaluation in the Puzzletron worker environment, choose the
route that matches the checkpoint and task:

| Route | Default smoke | Guide |
| --- | --- | --- |
| Text | `python -m examples.puzzletron.evaluation.text` runs eight IFEval and GSM8K samples with `lmms-eval`; the same command can prepare selected NeMo Evaluator task contracts. | [Text checkpoint evaluation](docs/checkpoint_evaluation.md) |
| Qwen 3.5 VLM | `python -m examples.puzzletron.evaluation.vlm.run` runs the pinned RealWorldQA and MMMU smoke suite. | [VLM checkpoint evaluation](docs/vlm_checkpoint_evaluation.md) |

The guides own installation, complete commands, suite or task selection,
runtime constraints, results, and troubleshooting for their respective routes.

## Configure a campaign

Use the maintained smoke and longer example above to learn the workflow before
creating a different campaign. The setup wizard is the customization path for
another model, dataset, search profile, or execution environment:

```bash
python examples/puzzletron/puzzletron_setup_v2.py \
  --defaults examples/puzzletron/configs/setup/defaults.example.yaml
```

The wizard is implemented by `puzzletron_setup_v2.py`. It inspects model
configuration metadata and writes validated smoke and production bundles. Each
bundle contains the same three inputs used above: experiment, runner, and
execution. For named width and depth search, setup derives the teacher hidden
size and layer count and stops on missing metadata instead of asking users to
edit generated YAML.

- Use [configuration and overrides](docs/configuration_overrides.md) to find
  the built-in configuration files, choose where campaign outputs are stored,
  or temporarily change experiment settings.
- Use the [Qwen VLM example](docs/qwen3p5_0p8b_vlm_smoke.md#change-the-example)
  to change measured architecture dimensions while keeping omitted dimensions
  at their teacher values.
- Use [Slurm configuration](docs/slurm_configuration.md) to change partitions,
  CPU-routed stages, log locations, and accepted compatibility fields.
- Use the [setup wizard guide](docs/setup_wizard.md) for profiles, generated
  files, non-interactive setup, and setup resume.

Run `--dry-run` after every configuration change. It resolves and validates
the experiment, runner, and execution files before any job is submitted.
Incompatible generated bundles are rejected with the setup-resume command that
regenerates both bundles; the [setup wizard guide](docs/setup_wizard.md#generated-files)
lists the compatibility checks.

## Operate and recover a campaign

See [run and recovery options](docs/orchestration_operations.md) for individual
stages, `--once`, non-interactive behavior, logging controls, security options,
execution strategies, saved run state, and recovery. Remote model code and
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
