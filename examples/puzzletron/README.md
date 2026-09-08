# Puzzletron v2

Puzzletron v2 creates resumable pruning campaigns that search model shapes,
validate the selected candidates, and optionally evaluate, benchmark,
materialize, or distill them. Start here to choose a supported route. Each
linked guide owns the complete commands and result checks for its task. The
maintained VLM smoke commands are repeated here to keep the first run linear;
the model guide remains their canonical reference.

## Table of contents

- [Start here: lifecycle smoke](#start-here-lifecycle-smoke)
- [Choose the next task](#choose-the-next-task)
- [Documentation map](#documentation-map)

## Start here: lifecycle smoke

Start with the maintained Qwen 3.5 0.8B VLM smoke. It exercises the complete
image-text lifecycle with bounded workloads inside one reusable Slurm allocation
that reserves two GPUs on a single node. Logical stages still have separate
attempts, logs, artifacts, and resume checks. Once it succeeds, the optional
longer example campaign uses the same environment, runner, orchestrator,
progress display, report, and resume command. Logical stage progress is emitted
in the reusable allocation log. That larger campaign takes longer than the
smoke; its duration depends on worker hardware, scheduler availability, cache
state, and the execution profile. It is intended for scheduled integration
validation, not routine smoke or presubmit use.

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

These accelerated profiles are qualified for a Slurm environment with
eight-GPU nodes: the smoke reserves two GPUs on one node, while the
representative campaign reserves all eight GPUs on one node. They are not
qualified for arbitrary Slurm sites. See [Slurm
configuration](docs/slurm_configuration.md#reusable-single-node-allocations)
for the assumptions and settings another site may need to adapt.

The three campaign inputs have separate responsibilities:

- The experiment file defines the model, data, and bounded smoke workload. Do
  not edit it for this first run.
- The runner file contains the site-specific scheduler and worker settings.
- The execution file selects reusable allocation and maps each logical stage
  onto a GPU lease or zero-GPU CPU execution inside the shared allocation.

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
EXECUTION=examples/puzzletron/configs/orchestration/qwen3p5_0p8b/execution.vlm_smoke.yaml
RUNNER=runner.slurm.yaml

python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full --dry-run
```

Review the resolved paths, CPU and GPU requests, worker image, mounts, and log
locations. This profile's dry-run shows one outer allocation and the logical
attempts that will execute inside it. Then run the same plan without
`--dry-run`:

```bash
python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full
```

### 5. Resume and inspect results

Run the launch command above again to recover an interrupted smoke or verify a completed one. It reattaches to a compatible active outer allocation. After Slurm reports that allocation failed or was cancelled, it starts a replacement unless a compatible worker result records clean completion or an uncancelled terminal stage failure. The replacement reruns only unfinished compatible work. A recorded cancellation and a final-report failure remain retryable; completed logical stages are not rerun. Changing the configuration or run root creates a new plan identity; compatible completed stages in the same run root are still not submitted again. Puzzletron stores its structured runtime state under `$PUZZLETRON_RUN_ROOT/orchestration/`.

The reusable-allocation watcher shows the outer Slurm state and allocation log
path. Follow that log to see the inner controller's completed/total stages,
queued and running stages, elapsed time, 30-second heartbeats, and native inner
units such as evaluation samples and measured rate. An ETA appears only after
progress provides a reliable denominator and the controller observes
throughput; otherwise it says `ETA unavailable`.

After the selected plan completes cleanly, `orchestrate.py` attempts to write the
final report to
`<puzzle-dir>/artifacts/campaign_report/campaign_report.html`. A report failure
does not invalidate completed logical stages, but it is recorded in the run
result and the command exits nonzero so the report can be retried. See
[run and recovery options](docs/orchestration_operations.md) for individual
stages, `--once`, logging controls, security options, and recovery details, or
[campaign reports](docs/campaign_reports.md) to regenerate and interpret a
report. For a failed or interrupted run, follow the actionable checks in
[run and recovery options](docs/orchestration_operations.md#progress-and-interruption).

The [Qwen VLM example guide](docs/qwen3p5_0p8b_vlm_smoke.md) lists the smoke's
expected lifecycle checks and explains how to interpret its bounded results.

### 6. Optional: run the longer example

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
candidate coverage, and distillation work. Its execution profile uses
one eight-GPU node and configured within-node candidate concurrency when
dependencies are ready, but does not introduce another operational path. Use
focused tests and the lifecycle smoke for routine development; reserve this
campaign for scheduled integration validation.

## Choose the next task

These guides own their model-specific configuration, launch commands, expected
artifacts, and interpretation limits:

| Model and route | Use it for |
| --- | --- |
| [Qwen 3.5 0.8B text smoke](docs/qwen3p5_0p8b_smoke.md) | Bounded text pruning, checkpoint reload, IFEval, serving, short KD, and resume |
| [Qwen 3.5 0.8B VLM smoke and campaign](docs/qwen3p5_0p8b_vlm_smoke.md) | Bounded lifecycle smoke and a broader illustrative multi-axis VLM campaign |
| [Qwen 3.5 4B VLM example](docs/qwen3p5_4b_vlm_example.md) | FFN search with opt-in lifecycle and longer campaign routes |

Run the smoke for a model in the same worker environment before starting its
larger campaign. A campaign guide may require more data, GPUs, walltime, or
manual approval than its smoke.

### Evaluate a checkpoint

Evaluation can run inside a post-MIP flow so candidate lineage and metrics stay
with the campaign. See [evaluate saved checkpoints](docs/post_mip_pipeline.md#evaluate-saved-checkpoints)
for that route.

For a standalone local Hugging Face checkpoint, choose the matching evaluator:

| Checkpoint | Guide | Owner of exact commands and outputs |
| --- | --- | --- |
| Text | [Text checkpoint evaluation](docs/checkpoint_evaluation.md) | `python -m examples.puzzletron.evaluation.text` and NeMo Evaluator preparation |
| Qwen 3.5 VLM | [VLM checkpoint evaluation](docs/vlm_checkpoint_evaluation.md) | Benchmark profiles, cache preparation, preflight, execution, and result interpretation |

Keep evaluator, task contract, dataset revision, sample selection, model
revision, generation settings, and judge identity with every result. Scores
from different evaluator paths or profiles are not interchangeable.

## Documentation map

The documentation is organized by user task. Follow links from this page
rather than searching configuration or implementation directories.

### Set up and prepare

- [Environment setup](docs/environment_setup.md) explains the controller and
  worker environments.
- [Worker image](docs/worker_image.md) owns image build, export, GPU check, and
  identity instructions.
- [Setup wizard](docs/setup_wizard.md) owns profiles, model and dataset inputs,
  advanced automation, setup resume, and generated files.

### Run and recover

- The [maintained campaign routes](#choose-the-next-task) own tracked
  model-specific commands and expected artifacts.
- [Run and recovery](docs/orchestration_operations.md) owns stage selection,
  dry-run behavior, progress, interruption, recovery, and execution records.

### Evaluate and read results

- [Text checkpoint evaluation](docs/checkpoint_evaluation.md) and
  [VLM checkpoint evaluation](docs/vlm_checkpoint_evaluation.md) own standalone
  evaluation.
- [Post-MIP pipelines](docs/post_mip_pipeline.md) owns evaluation and other
  processing attached to candidate lineage.
- [Campaign reports](docs/campaign_reports.md) owns report regeneration and the
  navigation boundary for retained results. Campaign and run leaves own their
  own measurements, provenance, reproduction evidence, and limitations.

### Configure and diagnose

- [Configuration and overrides](docs/configuration_overrides.md) covers config
  layering, output roots, and temporary experiment changes.
- [Slurm configuration](docs/slurm_configuration.md) covers partitions,
  CPU-routed stages, logs, and scheduler settings.
- [MIP runs](docs/mip_profiles.md) covers objectives, constraints, search
  spaces, variants, and homogeneous search.
- [Sanity validation](docs/sanity_validation.md) explains sorting, width
  ranking, physical slicing, warnings, and correctness failures.

### Understand and extend

- [Architecture](docs/v2_architecture.md) maps the campaign DAG and components
  to their implementation locations.
- [Legacy Nano campaign](docs/legacy_nano_campaign.md) documents the separate
  retained Nano workflow; it is not the default Puzzletron v2 route.
- To run a campaign with an agent, ask it to use
  [`running-puzzletron`](../../.agents/skills/running-puzzletron/SKILL.md) and
  provide the model, dataset, compute environment, search space, resource
  constraints, and required downstream stages.
