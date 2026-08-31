# Puzzletron v2

Puzzletron v2 helps you explore model architectures and find smaller, faster
variants that meet your quality and deployment goals. Guided setup creates
reproducible campaigns that find and compare candidates. Campaigns can
evaluate, benchmark, materialize, or distill those candidates.

## Table of contents

- [First campaign](#first-campaign)
- [Understand the campaign stages](#understand-the-campaign-stages)
- [Evaluate a checkpoint](#evaluate-a-checkpoint)
- [Configure a campaign](#configure-a-campaign)
- [Operate and recover a campaign](#operate-and-recover-a-campaign)
- [Extend Puzzletron](#extend-puzzletron)

## First campaign

The usual path is to prepare Puzzletron, generate a campaign, validate the
generated setup with a bounded run, and then launch the campaign. The same
command resumes compatible work after an interruption.

For an image-text walkthrough using Qwen 3.5 0.8B and the recommended
Nemotron-VLM dataset, follow the
[Qwen VLM pruning smoke](docs/qwen3p5_0p8b_vlm_smoke.md).

### 1. Prepare the environments

Create one lightweight Python environment for the setup wizard and the command
that launches campaigns:

```bash
python3 -m venv .venv-puzzletron
source .venv-puzzletron/bin/activate
python -m pip install -r examples/puzzletron/requirements-setup.txt
```

This environment creates campaign files and runs `orchestrate.py`. Model
conversion, training, evaluation, and benchmarking run in the worker
environment or container selected during setup. Prepare the
[worker environment](docs/environment_setup.md) before launching a campaign.

### 2. Generate a campaign

Start the guided setup with the repository defaults:

```bash
python examples/puzzletron/puzzletron_setup_v2.py \
  --defaults examples/puzzletron/configs/setup/defaults.example.yaml
```

Choose **Balanced pruning** for a first-generated text campaign. For the
maintained Qwen text route, select `Qwen/Qwen3.5-0.8B` and the recommended
Puzzle-KD v2 text dataset or an existing worker-visible dataset. For VLM, use
the VLM pruning smoke linked below. Guided setup can resolve the
same model, the recommended Nemotron-VLM v2 image-text dataset, and site
settings, but its generic pruning profile is not the VLM example used here.
Review the detected model, data modality, worker and scheduler settings, and
output directory.

The wizard reads model configuration, not model weights, and does not submit
jobs. It writes a bounded validation bundle under `smoke/`, a campaign bundle
under `production/`, and a generated `README.md` with the commands for both.
Users do not need to assemble or edit a separate smoke configuration. Run any
dataset preparation command in that generated README from the worker
environment before launch. For Qwen 3.5 0.8B, both bundles include
a final pinned student-versus-teacher downstream comparison that records
results without imposing an acceptance threshold. See the
[setup wizard guide](docs/setup_wizard.md) for profiles, hosted datasets, full
configuration mode, generated files, and setup resume.

For a Qwen 3.5 0.8B text pruning example, see the
[Qwen 3.5 0.8B campaign](docs/qwen3p5_0p8b_campaign.md). It searches two FFN
intermediate sizes, increases the scoring, serving, and distillation budgets,
and reuses the same downstream evaluation settings as the opt-in quality
comparison. The same guide includes an opt-in extended variant with hidden
width, attention, GDN, embedding-width, and depth pruning.

### 3. Validate the generated setup

Activate `.venv-puzzletron` and inspect the generated smoke plan:

```bash
PUZZLETRON_BUNDLE=/path/to/generated/campaign/smoke

python examples/puzzletron/orchestrate.py \
  --experiment "$PUZZLETRON_BUNDLE/experiment.yaml" \
  --runner "$PUZZLETRON_BUNDLE/runner.yaml" \
  --execution "$PUZZLETRON_BUNDLE/execution.yaml" \
  --stage full --dry-run
```

Review the stage list, worker paths, resources, and output directory.
`--stage full` runs every stage enabled by the generated validation experiment
in dependency order. Remove `--dry-run` to launch it. This bounded run is the
default setup check; detailed smoke limits and manually maintained smoke recipes
are documented for advanced use and reproducibility in the
[text](docs/qwen3p5_0p8b_smoke.md) and
[VLM](docs/qwen3p5_0p8b_vlm_smoke.md) guides.

### 4. Launch the campaign

After validation succeeds, change `smoke` to `production`, inspect that plan
with `--dry-run`, and launch it with the same command.

The experiment file defines the model and algorithm choices, the runner file
defines the worker environment, and the execution file says how each stage
runs. Keep all three together when launching or resuming a generated campaign.

### 5. Resume and inspect results

The experiment file calls the campaign output directory `puzzle_dir`.
`orchestrate.py` shows live progress and stores resume information under
`<puzzle-dir>/orchestration/`. Run the same command with the same three files to
recover a detached or interrupted campaign; completed compatible stages are not
submitted again.

After the selected plan completes cleanly, `orchestrate.py` attempts to write the
final report to
`<puzzle-dir>/artifacts/campaign_report/campaign_report.html`. A report failure
does not fail the completed campaign and is recorded in the run result. See
[run and recovery options](docs/orchestration_operations.md) for individual
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
[post-MIP pipeline guide](docs/post_mip_pipeline.md#evaluate-saved-checkpoints).

For standalone evaluation in the Puzzletron worker environment, choose the
route that matches the checkpoint and task:

| Route | Default smoke | Guide |
| --- | --- | --- |
| Text | `python -m examples.puzzletron.evaluation.text` runs eight IFEval and GSM8K samples with `lmms-eval`; the same command can prepare selected NeMo Evaluator task contracts. | [Text checkpoint evaluation](docs/checkpoint_evaluation.md) |
| Qwen 3.5 0.8B VLM | `python -m examples.puzzletron.evaluation.vlm.run` runs the pinned RealWorldQA and MMMU short suite. | [VLM checkpoint evaluation](docs/vlm_checkpoint_evaluation.md) |

The guides own installation, complete commands, suite or task selection,
runtime constraints, results, and troubleshooting for their respective routes.

## Configure a campaign

Generated bundles contain an experiment, runner, and execution file. The
default route above is sufficient when their generated values match your model
and infrastructure.

- Use [configuration and overrides](docs/configuration_overrides.md) to find
  the built-in configuration files, choose where campaign outputs are stored,
  or temporarily change experiment settings.
- Use the
  [Qwen campaign extension steps](docs/qwen3p5_0p8b_campaign.md#change-the-search-dimensions)
  to add a measured architecture dimension while keeping omitted dimensions
  at their teacher values.
- Use [Slurm configuration](docs/slurm_configuration.md) to change partitions,
  CPU-only stages, log locations, and accepted compatibility fields.
- Use the [setup wizard guide](docs/setup_wizard.md) to change profiles,
  datasets, generated files, or setup automation.

Run `--dry-run` after every configuration change. It resolves and validates
the experiment, runner, and execution files before any job is submitted.

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
