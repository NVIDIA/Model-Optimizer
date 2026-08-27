# Puzzletron v2

Puzzletron v2 helps you explore model shapes and select a smaller, faster
variant against your quality and deployment goals. Its guided setup creates a
reproducible, resumable campaign that compares candidates and can optionally
distill the selected model.

## Table of Contents

- [Start here](#start-here)
- [Installation](#installation)
- [Setup wizard](#setup-wizard)
- [Evaluate a checkpoint](#evaluate-a-checkpoint)
- [Run with an agent](#run-with-an-agent)
- [Configuration](#configuration)
- [Experiment overrides](#experiment-overrides)
- [Slurm configuration](#slurm-configuration)
- [Qwen 3.5 smoke test](#qwen-35-smoke-test)
- [Run a campaign](#run-a-campaign)
- [Controller operations](#controller-operations)
- [MIP runs](#mip-runs)
- [Post-MIP pipelines](#post-mip-pipelines)
- [Sanity validation](#sanity-validation)
- [Reports](#reports)
- [Legacy Nano campaign](#legacy-nano-campaign)
- [Architecture](#architecture)

## Start here

- **New campaign:** complete the [installation](#installation), then use the
  [setup wizard](#setup-wizard) to generate validated smoke and production
  bundles.
- **Generated campaign:** complete the [installation](#installation), then
  [run the campaign](#run-a-campaign) with its generated bundle.
- **Checkpoint evaluation:** use [Evaluate a checkpoint](#evaluate-a-checkpoint)
  for a local model without creating or running a pruning campaign.
- **Agent-assisted campaign:** follow [Run with an agent](#run-with-an-agent)
  with your model, data, compute environment, and deployment goals.
- **Existing results:** see [Reports](#reports) to regenerate a campaign report
  or inspect the retained examples.

## Installation

See [environment setup](docs/environment_setup.md) for worker containers,
pinned CUDA and PyTorch packages, patched dependencies, model-specific kernels,
bare-metal environments, and verification.

Use a lightweight environment for the setup wizard and controller:

```bash
python3 -m venv .venv-puzzletron-control
source .venv-puzzletron-control/bin/activate
python -m pip install \
  -r examples/puzzletron/requirements-setup.txt \
  -r examples/puzzletron/requirements-orchestrator.txt
```

GPU workers use the environment or container declared in the generated runner
file. Prepare that environment before launch, then run the smoke bundle first.

### Standalone runtime image

The repository-owned [`Dockerfile`](Dockerfile) builds the validated Qwen and
Nemotron runtime with ModelOpt, the patched vLLM fork, AutoModel, AIPerf,
flash-linear-attention, Mamba, causal-convolution, and grouped-GEMM installed
against one PyTorch and CUDA environment. The
[environment manifest](ci_environment.json) records the immutable CUDA base,
exact VCS revisions, verified core package versions, and CUDA architectures.
The Dockerfile is the sole installation recipe for this environment.

The Mamba package is built from the exact official `state-spaces/mamba` release
commit. Its release metadata pins TileLang 0.1.8, while the pinned vLLM revision
requires 0.1.9, so the build applies a repository-owned compatibility patch to
Mamba's dependency metadata. The manifest records the upstream commit and
patch checksum, and the final `pip check` rejects an inconsistent environment.

The grouped-GEMM revision used by the Nemotron path only declares CUDA
architectures through Hopper, so its build is recorded separately as
`8.0;8.6;9.0`. The remaining runtime extensions retain the broader architecture
set in the manifest.

Build the image from the repository root and record the ModelOpt revision in
its OCI metadata:

```bash
docker build \
  --platform linux/amd64 \
  --file examples/puzzletron/Dockerfile \
  --build-arg MODELOPT_REVISION="$(git rev-parse HEAD)" \
  --tag modelopt-puzzletron-runtime:local \
  .
```

The build verifies package versions, immutable VCS sources, CUDA compatibility,
and imports without requiring a GPU. Run the same checks again with the
standalone verifier:

```bash
docker run --rm modelopt-puzzletron-runtime:local \
  python /opt/puzzletron/verify_image_environment.py \
    --environment /opt/puzzletron/ci_environment.json \
    --profile runtime
```

Mount only the model, data, and result paths needed by a run:

```bash
export PUZZLETRON_WORKSPACE=/absolute/path/to/workspace
docker run --gpus all --ipc=host --rm -it \
  -v "${PUZZLETRON_WORKSPACE}:/workspace" \
  -e PUZZLETRON_RUN_ROOT=/workspace/results \
  modelopt-puzzletron-runtime:local
```

CI uses the same full image. A pull-request checkout is mounted over the baked
source and installed with `--no-deps`, so CI tests new ModelOpt code without
changing the image's third-party environment. The image workflow also runs the
focused lifecycle test in that overlay mode.

This change defines and validates the image but does not publish it. Image
publication is a separate trusted workflow that will push the verified build
to an approved registry and expose its immutable digest. CI, cluster jobs, and
external users should consume that same digest instead of rebuilding the
environment independently.

Successful image construction proves the environment contract only. Exact
vLLM runtime-stat replay remains a separate GPU workload whose cache identity,
hardware, workload, and measured endpoints must be recorded with the campaign;
image-build validation does not make a performance claim.

## Setup wizard

See the [setup wizard guide](docs/setup_wizard.md) for profiles, hosted dataset
handling, full configuration mode, generated files, and resuming an interrupted
setup.

The setup wizard reads a local checkpoint or Hugging Face model configuration
and generates validated smoke and production bundles. It does not load model
weights.

Start the wizard with the repository's example defaults file:

```bash
python examples/puzzletron/puzzletron_setup_v2.py \
  --defaults examples/puzzletron/configs/setup/defaults.example.yaml
```

Choose **Balanced pruning** for a first campaign, review the detected model and
infrastructure settings, and select an output directory. The generated
`README.md` contains any dataset preparation command and the exact paths for
the smoke and production bundles. The wizard prepares files but does not submit
jobs.

## Evaluate a checkpoint

See [checkpoint evaluation](docs/checkpoint_evaluation.md) for task selection,
full evaluation, result locations, and model-detection overrides.

Basic evaluation is independent of MIP and the campaign DAG. In the Puzzletron
worker environment, run any compatible local Hugging Face checkpoint directly:

```bash
python examples/puzzletron/evaluate_lmms_checkpoint.py \
  --checkpoint /path/to/checkpoint \
  --output-dir /path/to/results/checkpoint-smoke
```

The default one-GPU smoke evaluates eight samples each from IFEval and GSM8K.
Qwen 3.5 checkpoints are configured automatically. For options not covered by
the convenience command, append `--lmms-eval-args` followed by the native
lmms-eval options.

## Run with an agent

The canonical agent workflow is
[`running-puzzletron`](../../.agents/skills/running-puzzletron/SKILL.md). Ask an
agent to use that skill and provide the model, dataset, compute environment,
search space, resource constraints, and required downstream stages. For
example:

```text
Use .agents/skills/running-puzzletron/SKILL.md to run the Puzzletron campaign
at examples/puzzletron/configs/families/nemotron3/nano_30b_a3b_bf16/runs/default.yaml.
Validate the smoke path first, execute the enabled DAG, resume compatible
artifacts, and regenerate and verify the report after every completed stage.
```

`.agents/` is the source of truth. Agent-specific paths such as
`.claude/skills/running-puzzletron` are compatibility symlinks and should not
be edited separately.

## Configuration

Configs use Hydra composition:

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

Site-specific paths can be overridden without editing the checked-in config:

```bash
export PUZZLETRON_RUN_ROOT=/shared/puzzle_runs/my_campaign
```

`PUZZLETRON_RUN_ROOT` is a convenience used by the checked-in experiment YAMLs
to resolve `puzzle_dir`. Generated bundles write their chosen `puzzle_dir`
directly. In both cases, `puzzle_dir` is the canonical location for artifacts,
manifests, controller state, and logs unless `runner.slurm.log_dir` relocates
attempt logs.

## Experiment overrides

See [experiment overrides](docs/configuration_overrides.md) for temporary
changes without editing the checked-in YAML.

Overrides can select another run root, adjust a campaign value, or change one
stage while preserving the source configuration. Validate the resolved config
before launch so misspelled or misplaced fields fail at the command boundary.

## Slurm configuration

See [Slurm configuration](docs/slurm_configuration.md) for partition lists,
CPU-only stages, log directories, and accepted compatibility fields.

Use the checked-in runner and execution examples as templates, replace their
site placeholders, and inspect the plan with `--dry-run` before launch. Runner
files own infrastructure; execution files own per-stage strategy and resource
selection.

## Qwen 3.5 smoke test

See the [Qwen 3.5 0.8B smoke guide](docs/qwen3p5_0p8b_smoke.md) for the
one-GPU route, dry run, and manual GPU acceptance test.

This focused campaign checks the MIP path on a small public checkpoint before
larger model or cluster runs.

## Run a campaign

Activate the control environment and run the generated smoke bundle first. The
smoke run checks the worker environment and campaign wiring before the larger
production run:

```bash
PUZZLETRON_BUNDLE=/path/to/generated/campaign/smoke

python examples/puzzletron/orchestrate.py \
  --experiment "$PUZZLETRON_BUNDLE/experiment.yaml" \
  --runner "$PUZZLETRON_BUNDLE/runner.yaml" \
  --execution "$PUZZLETRON_BUNDLE/execution.yaml" \
  --stage full
```

After the smoke campaign succeeds, change `smoke` to `production` and run the
same command. Add `--dry-run` before either launch to inspect the plan without
submitting jobs.

## Controller operations

See [controller operations](docs/orchestration_operations.md) for individual
stages, non-interactive behavior, logging options, execution strategies,
recovery, and controller records.

Remote model code and AIPerf v0.11 online tokenizer resolution are disabled by
default. Enable remote code only for a trusted model source. The tokenizer
compatibility option permits the AIPerf child process to resolve its tokenizer
online even when the surrounding campaign is configured for offline loading.

The controller shows live progress and keeps resume state under
`${puzzle_dir}/orchestration/`. Press `q` or Ctrl-C in an interactive terminal
to cancel, detach, or continue. Run the same command again to recover a detached
campaign.

## MIP runs

See [MIP runs](docs/mip_profiles.md) for variants, solution pools, objectives,
resource constraints, workload measurements, and homogeneous search.

Named profiles let one campaign compare candidate architectures against
different parameter, runtime, or memory goals without duplicating the earlier
importance and scoring stages.

## Post-MIP pipelines

See [post-MIP pipelines](docs/post_mip_pipeline.md) for candidate evaluation,
filtering, materialization, AIPerf, and distillation.

These downstream nodes turn selected MIP solutions into evaluated or
materialized checkpoints and can continue through serving measurements and
global distillation.

## Sanity validation

See [sanity validation](docs/sanity_validation.md) for correctness checks,
ranking warnings, comparison controls, tolerances, and qualification guidance.

Sorting and slicing equivalence failures are correctness errors. Ranking
quality misses are warnings unless strict warning handling is enabled.

## Reports

See [campaign reports](docs/campaign_reports.md) for cache controls and the
evidence status of retained example reports.

After the selected plan completes cleanly, the v2 orchestrator generates the
final campaign report through the configured runner. Reporting is nonfatal to
the completed campaign, but a failed report attempt is recorded in the
controller result. The generator is read-only with respect to model artifacts
and includes valid partial results without marking their stage complete.

Regenerate without rerunning model work:

```bash
python examples/puzzletron/generate_campaign_progress_report.py \
  --puzzle-dir /shared/puzzle_runs/my_campaign \
  --model-name 'My model'
```

Open
`<puzzle-dir>/artifacts/campaign_report/campaign_report.html` locally.

## Legacy Nano campaign

See the [legacy Nano campaign](docs/legacy_nano_campaign.md) for the separate
online evaluation and finalist-materialization workflow used by the checked-in
Nano configuration.

That configuration uses `mode: online_solutions`; it does not use the generated
campaign DAG's integrated evaluation and materialization nodes.

## Architecture

See the [v2 architecture](docs/v2_architecture.md) for the stage registry,
campaign DAG, scheduler-neutral control plane, and maintainer guidance.

The experiment config owns model and algorithm semantics, the runner owns the
worker environment, and the execution config owns per-stage orchestration.
