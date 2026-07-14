# Puzzletron

Puzzletron searches for smaller, faster variants of pretrained language and multimodal
models while preserving a reproducible path from the teacher checkpoint to realized,
evaluated, and optionally distilled models. The current implementation is an
artifact-driven DAG: each stage consumes immutable upstream artifacts, publishes a
manifest, can resume independently, and contributes to one cumulative HTML report.

## Start Here

For a new model, use the [running-puzzletron skill](../../PUZZLETRON_SKILL.md). It guides
model-aware intake, source inspection, descriptor/backend selection, pruning-axis
admission, multimodal and MTP handling, full-coverage smoke validation, production config
generation, execution, resume, and reporting.

The skill must produce an actual runnable config bundle and exact commands after the
questionnaire; it is not only a review checklist. It also requires physical pruning to be
the ground truth for every dynamic slicing implementation.

For NVIDIA-internal operation, the ignored `nv-internal/sepehr_scripts.md` contains the
long-form step-by-step command sequence and `nv-internal/CLUSTER_GUIDE_NV.md` records the
current cluster behavior. These files are intentionally not part of the public Git tree.
The shorter local pointer is preserved at
`nv-internal/examples/puzzletron/sepehr_scripts.md`.

## Installation

Create one environment whose PyTorch, CUDA, ModelOpt, AutoModel, vLLM, and AIPerf builds
are ABI-compatible. From the repository root:

```bash
source .venv/bin/activate
python -m pip install -e .
python -m pip install -e ../vllm
python -m pip install -e ../aiperf
python -m pip install -e ../Automodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

Record package revisions and `torch.version.cuda` in the experiment. Revalidate imports
and one GPU forward after rebuilding any compiled sibling.

## What v2 Adds

The v2 branch is more than a new runner. Its reusable additions include:

- content-addressed identities, manifests, artifact inventory/import/coverage, and
  acceptance-based resume;
- scheduler-neutral stage graph and isolated stage workers;
- family/model capability registries, generic decoder contracts, HF conversion, native
  AutoModel descriptors, and AnyModel vLLM export;
- fixed, padded, packed, and multimodal batch contracts with sample hashing;
- coordinated activation passes, sorted teachers, reverse-sort and physical-slice sanity,
  hidden-width/PLE support, and typed attention/FFN/MoE/Mamba/GDN surgery;
- checkpointed elastic bypass with per-DP architecture observations and subblock boundaries;
- iterative depth trajectories, block/subblock candidate libraries, sparse sampling,
  shardable vLLM statistics, and distributed replace-one evaluation;
- multi-profile MIP, content-addressed solution registries, physical realization, exact
  evaluation, AIPerf sweeps, and post-distillation evaluation;
- full-vocabulary chunked/flash CE and KD objectives, MTP loss separation, VLM-aware global
  KD, scenario grids, and tournament selection;
- an artifact-driven cumulative HTML report with a navigable DAG, per-stage warnings,
  granularity-aware sections, interactive bypass/scoring plots, and partial-run visibility.

See the repository map below for the source ownership of each subsystem.

## Configuration

All supported configurations live under [`configs/clean/`](configs/clean/):

```text
configs/clean/
├── base.yaml                  # scheduler-neutral defaults and stage contracts
├── families/<family>/        # family defaults
│   └── <model>/model.yaml     # checkpoint, descriptor, axes, model geometry
├── recipes/                   # TP/CP/PP/EP/DP execution recipes
├── campaigns/                 # reusable multi-model campaign definitions
├── my_paths.example.yaml      # template for machine-local paths
└── my_paths.yaml              # ignored local overlay
```

An experiment YAML composes the base, family, model, topology recipe, and experiment
overrides. Keep checkpoint, dataset, container, account, and artifact-root values in the
ignored `my_paths.yaml`; committed configs must be portable.

Important configuration choices are independent:

- Data modality and layout: text or supported media combinations; fixed, padded, or
  packed variable-length batches.
- Per-stage granularity: depth, bypass, vLLM statistics, and replace-one scoring each
  choose block or subblock semantics independently.
- Search domains and constraints: legal axis values, depth scenarios, runtime/memory/
  parameter budgets, and numbers of models to evaluate or distill.
- Execution topology: TP, CP, PP, EP, DP, sequence parallelism, microbatch, and global
  batch. Validate the backend mesh rather than multiplying labels blindly.

Granularity is a per-component contract, not a campaign-wide switch:

| Component | Default | Meaning |
|---|---|---|
| Depth importance | `subblock` | Rank removable attention/FFN/typed sublayers independently |
| vLLM statistics | `block` | Measure full block combinations unless subblock additivity is requested |
| Candidate statistics | `block` | Build cost records at the configured library unit |
| Replace-one scoring | `block` | Score one block candidate; `subblock` scores one changed sublayer at a time |
| Bypass | `block` | Sample and train complete block candidates; `subblock` isolates local sublayer losses |

`Build Block Library` remains the public stage name at either granularity. Artifacts and
report labels record the actual per-stage setting; changing one component does not silently
change the others.

Parse the final config with the same loader used by the runner before allocating GPUs:

```bash
python - <<'PY'
from modelopt.torch.puzzletron.pipeline_config import pipeline_config_from_path

cfg = pipeline_config_from_path("path/to/experiment.yaml")
print(cfg["experiment"]["dir"])
PY
```

## Pipeline DAG

The authoritative registry is
[`stages/graph.py`](../../modelopt/torch/puzzletron/stages/graph.py). Required and optional
stages, dependencies, display names, distributed execution, and completion artifacts are
defined there.

```text
Convert Checkpoint
├── Tokenize Data
│   ├── Depth Importance Estimation (optional)
│   └── Width Importance Estimation
│       └── Sort Checkpoint
│           ├── Sort Sanity Check (optional)
│           ├── Width Sanity Check (optional)
│           ├── Slicing Sanity Check (optional)
│           └── Bypass Sanity Check (optional)
│               └── Bypass (optional; block or subblock)
│                   └── Build Block Library
│                       └── Replace-one-block/subblock Scoring
└── vLLM Stats (optional; block or subblock)

vLLM Stats + Depth Importance + Replace-one Scoring
└── MIP Search
    ├── AIPerf (optional)
    └── Zero-shot Evaluation (optional)
        └── Global Distillation Sanity Check (optional)
            └── Global Distillation (optional)
                └── Post Distillation Evaluation (optional)
```

The graph is not a command-order recommendation for one giant allocation. Run independent
branches concurrently when they have disjoint writers, and use the smallest appropriate
resource topology for each stage.

## Running the Pipeline

Run all enabled stages sequentially in dependency order:

```bash
python examples/puzzletron/main.py \
  --config path/to/experiment.yaml \
  --stage full \
  --gpus-per-node 8
```

Run one stage:

```bash
python examples/puzzletron/main.py \
  --config path/to/experiment.yaml \
  --stage width_importance \
  --gpus-per-node 8
```

Apply a resolved config override without editing the YAML:

```bash
python examples/puzzletron/main.py \
  --config path/to/experiment.yaml \
  --stage bypass_sanity \
  --override bypass.granularity=subblock
```

Use `--force` only when intentionally invalidating a valid completion marker. Normally the
runner skips a stage only when its marker, semantic config, upstream identities, and
required outputs agree.

The orchestrator launches a fresh local worker for each stage so distributed state and GPU
memory cannot leak across stages. For an externally launched distributed job, run exactly
one distributed stage. The launcher must establish the process group, for example:

```bash
torchrun --nnodes="$NNODES" --nproc-per-node="$GPUS_PER_NODE" \
  --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  examples/puzzletron/main.py \
  --config path/to/experiment.yaml \
  --stage replacement_scoring \
  --gpus-per-node "$GPUS_PER_NODE"
```

On Slurm, map one distributed model task per node or launch explicit independent workers;
on bare metal, use passwordless SSH, shared storage at identical paths, deterministic rank
mapping, per-host logs/PIDs, and complete peer cleanup after a failure.

The supplied container launchers take machine-local values from the environment rather
than embedding filesystem paths:

```bash
export PUZZLETRON_IMAGE="path/to/container.sqsh"
export PUZZLETRON_CONTAINER_MOUNTS="shared:shared"
export PUZZLETRON_SETUP_ENV="path/to/optional-setup-env.sh"  # optional
export PUZZLETRON_RUN_ROOT="puzzle_runs"                     # optional
```

Keep concrete values in an ignored machine-local script or scheduler export file.

## Stage Correctness

### Conversion and data

Conversion must preserve teacher outputs, model-specific config, tokenizer/processor
assets, tied-weight semantics, and backend loadability. Tokenization must preserve sample
identity, padding/loss masks, packed document boundaries, modality fields, and MTP target
boundaries.

### Width importance, sorting, and slicing

Inventory every distinct layer implementation before defining pruning axes. Each admitted
axis needs a dynamic implementation, physical materialization, state-dict conversion, cost
model, backend compatibility, and equivalence case.

Keep the three diagnoses separate:

- Sort sanity compares full-size original, sorted, and reverse-sorted checkpoints.
- Width sanity checks whether the learned ranking improves reduced candidates.
- Slicing sanity compares dynamic sorted, dynamic unsorted, dynamic reverse, and physically
  materialized sorted variants against the original teacher.

Physical materialization is the oracle. A mismatch is a bug until localized; do not change
physical pruning to match a mask.

### Bypass

Run fixed-smallest and diverse-resampling sanity checks before a long nested bypass. The
fixed candidate should overfit clearly; the diverse trend should decrease despite scatter.
When DP ranks sample distinct architectures, save rank, step, canonical architecture hash,
human-readable config, parameter ratio, sample identity, and component losses for every
observation.

### Expensive immutable stages

Depth importance, vLLM statistics, and replace-one scoring must use shard manifests and
resume only missing identities. They should remain usable when worker count/topology changes
if their artifact schema is topology-independent. Never delete or rerun complete expensive
shards merely to regenerate a report.

### MIP, evaluation, AIPerf, and KD

MIP solutions must reference exact score/runtime/depth inputs and satisfy the declared
constraints after physical realization. Include the teacher in zero-shot evaluation.
AIPerf should sweep meaningful concurrency/topology values and verify successful requests,
token lengths, and instruct templates.

Global KD sanity must overfit before a real run. Report `main_ce`, `mtp_ce`, `main_kd`, and
`mtp_kd` separately and compute their configured weighted sum. Shifted MTP targets must not
cross padding, packed-document, or modality boundaries. Validate interruption/resume before
trusting a long job.

## Resuming and Checkpointing

Long-running stages should publish before the scheduler deadline. A training checkpoint is
complete only when model, optimizer, scheduler/scaler, dataloader cursor, global step, all
RNG states, topology metadata, architecture observations, and a completion marker agree.

Write checkpoints transactionally:

1. Write all shards to a temporary transaction directory.
2. Validate expected shards and identities.
3. Publish the completion marker atomically.
4. Update `latest` only after publication.
5. Quarantine incomplete transactions and resume the newest complete checkpoint.

Use normal dependency-on-success for DAG transitions. A continuation after timeout or
failure is safe only for a stage whose resume contract has been validated.

## Report

Every worker calls
`diagnostics.campaign_progress_report.generate_campaign_progress_report` and refreshes the
cumulative report from artifacts on disk. The stable HTML path is:

```text
<experiment-dir>/artifacts/campaign_report/campaign_report.html
```

Regenerate it without rerunning a stage:

```bash
python - <<'PY'
from modelopt.torch.puzzletron.diagnostics.campaign_progress_report import (
    generate_campaign_progress_report,
)

result = generate_campaign_progress_report("puzzle_runs/experiment", model_name="Model name")
print(result)
PY
```

The generator reads existing artifacts only; it must not mutate or invalidate completed
stage data.

The report contains:

- experiment identity and resolved important config;
- pipeline DAG with completed, pending, and disabled nodes;
- artifact provenance and stage warnings;
- sorting, width, slicing, bypass, depth, runtime, scoring, MIP, evaluation, AIPerf, and KD
  sections whenever their artifacts exist;
- partial long-running bypass/KD observations without erasing earlier results.

Warnings should appear beside the affected values. A warning does not make an artifact
incomplete, but it must remain visible and explain the expected invariant.

## Adding a New Model

Use the Puzzletron skill instead of copying a nearby descriptor blindly. At minimum:

1. Resolve the exact checkpoint and inspect every distinct HF layer implementation.
2. Read matching vLLM and AutoModel implementations when they exist.
3. Select AutoModel only when the unpruned model is already natively supported; otherwise
   use the HF-native Puzzletron path.
4. Admit only axes supported by all required constructors, kernels, caches, export paths,
   and physical materializers.
5. Exercise every selected modality/layout and topology path in a tiny full-coverage smoke.
6. Deliberately interrupt and resume bypass and Global KD.
7. Produce a resolved production config bundle and exact direct/Slurm/SSH commands before
   requesting approval for expensive compute.

## Debugging

Debug in dependency order:

1. Confirm resolved config, revisions, environment, and artifact identity.
2. Inspect one processor-native batch and all masks/media/packed metadata.
3. Run one unpruned forward under the exact backend and topology.
4. Compare dynamic and physical outputs at the first changed module boundary.
5. Read every rank log and diagnose the earliest failure before changing resources.
6. Validate shard/checkpoint manifests before resuming.
7. Regenerate the report and verify its node states follow artifacts and active config.

Common failures include wrong PP schedules, TP without required sequence parallelism,
unequal CP token reductions, dropped multimodal keys, packed targets crossing documents,
incomplete checkpoints presented as `latest`, and allocated GPUs with no active workers.

## Repository Map

- [`main.py`](main.py): scheduler-neutral orchestrator and isolated worker launcher.
- [`modelopt/torch/puzzletron/stages/`](../../modelopt/torch/puzzletron/stages): stage
  graph and handlers.
- [`modelopt/torch/puzzletron/anymodel/`](../../modelopt/torch/puzzletron/anymodel):
  model descriptors, conversion, and capability registry.
- [`modelopt/torch/puzzletron/plugins/automodel/`](../../modelopt/torch/puzzletron/plugins/automodel):
  native AutoModel hooks, scoring, evaluation, bypass, and KD integration.
- [`modelopt/torch/puzzletron/pruning/`](../../modelopt/torch/puzzletron/pruning):
  dynamic candidates, sorting, physical materialization, and axis surgery.
- [`modelopt/torch/puzzletron/diagnostics/`](../../modelopt/torch/puzzletron/diagnostics):
  sanity aggregation and cumulative HTML report.
- [`modelopt/torch/puzzletron/replacement_library/`](../../modelopt/torch/puzzletron/replacement_library):
  candidate library and score composition.
- [`modelopt/torch/puzzletron/distillation/`](../../modelopt/torch/puzzletron/distillation):
  global KD datasets, losses, recipes, checkpointing, and publication.
- [`tests/unit/torch/puzzletron/`](../../tests/unit/torch/puzzletron): focused contracts
  for pipeline stages and model-independent behavior.

Keep public examples generic. Site-specific scheduler commands, personal experiment notes,
and historical campaign debugging belong under ignored `nv-internal/`, not in this tree.
