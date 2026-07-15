---
name: running-puzzletron
description: Use when onboarding a new LLM or multimodal model to Puzzletron, running or resuming a Puzzletron pruning campaign, adding model descriptors or pruning axes, validating activation hooks and physical slicing, collecting vLLM statistics, running MIP/evaluation/AIPerf/distillation, or debugging Puzzletron on Slurm or SSH-managed bare metal.
---

# Run Puzzletron

## What This Skill Does

This skill turns a model checkpoint and user requirements into a complete,
reproducible Puzzletron pruning campaign. It:

- inspects the model's actual source code before proposing anything;
- asks two short rounds of targeted questions (infrastructure, then search);
- generates runnable smoke and production config bundles with exact commands;
- validates the setup end-to-end with a tiny smoke campaign;
- executes the pipeline DAG, handles Slurm or bare-metal scheduling, resumes
  interrupted stages, and keeps the HTML report current.

If the model is not yet registered in Puzzletron, the skill adds the descriptor
and pruning axes as part of the workflow. If it is already supported, the skill
skips straight to campaign configuration.

If the user asks only for planning or documentation, produce the requested
files and stop. Do not launch any campaign work without explicit authorization.

## Core Contract

1. Ask the two-phase intake below; do not invent consequential campaign choices.
2. Inspect model source before proposing axes or entering Phase 2.
3. Admit only axes that hold through training, physical export, and all required
   inference backends.
4. Use physical materialization as the ground truth for dynamic slicing — a
   mismatch is a bug, not a tolerance question.
5. Run a tiny full-coverage smoke campaign and deliberately test resume.
6. Present one explicit readiness gate before expensive work.
7. Produce concrete smoke and production config bundles plus exact commands.
8. Execute the stage DAG from `stages/graph.py`; parallelize independent nodes.
9. Regenerate the cumulative HTML report after every completed stage.
10. Preserve or quarantine artifacts. Never silently overwrite incompatible work.

## Phase 1: Infrastructure and Goals

Ask these as one grouped message. Explain each item in plain language if the
user seems unfamiliar with Slurm, containers, or distributed training.

### Checkpoint and Data Access

- **Exact checkpoint URI or local path** — HuggingFace model ID or an absolute
  path on shared storage. Ask for the exact revision or commit hash when
  reproducibility matters.
  - *Why it matters:* Puzzletron pins the teacher identity at the start; a
    mismatched checkpoint invalidates all downstream artifacts.
- **Tokenizer/processor revision** — Usually the same as the model revision.
  Ask explicitly if the user is mixing a custom tokenizer with a base model.
- **Remote code trust policy** — Whether `trust_remote_code=True` is acceptable.
  Required for some models; a security decision the user must make consciously.
- **Dataset path/ID and access** — Where the calibration data lives and whether
  it requires a license token or special download step.

### Experiment Logistics

- **Experiment ID** — A short label (no spaces) used to name the artifact
  folder. Example: `llama3-8b-prune-v1`.
- **Artifact root** — Shared storage path where outputs will be written.
  Must be accessible from every worker node.
- **Existing artifacts policy** — If prior runs exist at this ID: resume, import
  from an alternate path, archive, or treat as untouched? Resuming is the
  default and the safest choice.

### Execution Environment

- **Scheduler: Slurm or bare metal?**
  - *Slurm:* ask for the cluster account/partition, available node types, GPU
    count per node, wall-time limits, and queue policy. Read the local
    `nv-internal/CLUSTER_GUIDE_NV.md` if present.
  - *Bare metal (SSH):* ask for hostnames, slots per host, SSH user,
    rendezvous host/port, and shared-storage root path. Require passwordless
    SSH and identical paths on every host.
- **Container or host Python?** — Enroot/Docker container image path, or a
  host-installed virtual environment? Both are supported; clarify mounts.
  - Example: `container at /lustre/containers/nemo.sqsh` or
    `venv at /home/user/repos/modelopt/.venv`.
- **Sibling checkout locations** — Where vLLM, AIPerf, and AutoModel are
  installed on the cluster. All four siblings must be in the same Python
  environment (same PyTorch/CUDA build).
- **Code change policy** — Whether changes to Puzzletron, AutoModel, or the
  vLLM fork are in scope for this campaign.

### Resources and Goals

- **Available nodes, GPU type, GPU count, and wall-time per job** — Used to
  size the smoke and production configs and to schedule appropriately.
- **Storage capacity for artifacts** — Large models can produce hundreds of GB
  of bypass/scoring artifacts.
- **Deadline and acceptable cost** — Informs whether to run all optional stages
  or a lean path.
- **Required final outcomes** — Which of the following does the user want?
  - Search only (MIP solutions with scores)
  - Zero-shot evaluation of top solutions
  - AIPerf throughput benchmarks
  - Bypass for richer scoring data
  - Global distillation of the best solution
  - Post-distillation evaluation
  - The complete DAG

## Inspect Before Asking Model-Specific Questions

Resolve the exact checkpoint revision and read every distinct computational
layer in the HuggingFace implementation: embeddings, attention variants,
MLP/MoE/expert layers, recurrent/SSM layers, normalization, multimodal towers,
routers, language heads, and MTP heads.

For each distinct layer type, also read (in order):
1. The HuggingFace implementation used by this exact checkpoint.
2. The vLLM implementation, if the unpruned model is already supported there.
3. The AutoModel implementation, if present.
4. Existing Puzzletron descriptors, hooks, materializers, and cost models.

Build an implementation inventory documenting, for each layer type:
- semantic candidate axes and legal value domains;
- coupled dimensions and alignment/divisibility constraints;
- HF, AutoModel, and vLLM constructor/runtime support;
- TP/CP/PP/EP implications;
- dynamic hook, physical materialization, export, and cost-accounting status;
- accepted axes, rejected axes, and evidence for each decision.

## Select the Model Path

Use an **AutoModel descriptor** only when AutoModel already natively supports
the unpruned model. Otherwise use Puzzletron's HF-native path.

Do not add native AutoModel or vLLM support for a previously unsupported base
model as part of this workflow. For an already-supported model, sibling fixes
are allowed when needed to make admitted pruned configurations work through the
existing AnyModel path.

If the unpruned model runs in vLLM, every admitted pruned configuration must
also run in the repository's AnyModel vLLM fork. Exclude axes that would
require a new model-specific vLLM constructor.

## Phase 2: Model-Aware Campaign Questions

Present only options that are valid for the inspected model. Group questions
by topic. Explain the trade-offs briefly for non-obvious choices.

### Data Lanes and Layouts

- **Which input modalities to include?** — Text only, text+image, text+audio,
  text+video, or combinations. Ask independently for each supported lane.
  - *Why it matters:* Each lane has separate importance statistics, bypass
    losses, and evaluation metrics. Omitting a lane may degrade the pruned
    model's performance on that modality.
- **Real dataset and split for each lane** — calibration samples, evaluation
  samples, license/access constraints, max sequence length, and sampling weights
  when mixing lanes.
- **Batch layout for each lane:**
  - *Fixed:* all examples shaped to the same length. Simple and predictable,
    but potentially wasteful.
  - *Padded:* variable examples batched to local max with attention masks.
    Conventional; adds padding overhead.
  - *Packed:* multiple documents share one sequence with explicit boundaries.
    Most GPU-efficient; requires careful handling of masks, targets, and
    multimodal metadata.
- **Chat/instruction template policy** — use the model's native instruct
  template for all evaluation and serving. Do not substitute generic role tags.

### Search and Granularity

- **Candidate ranges per axis** — for each admitted axis (e.g. `num_heads`,
  `head_dim`, `ffn_dim`, `num_experts`), what is the minimum value the user
  is willing to accept?
  - *Why it matters:* narrower ranges mean faster MIP but may miss good
    solutions; wider ranges need more scoring compute.
- **Maximum depth removals** — how many transformer layers the search may
  remove. Zero means depth search is disabled.
- **Bypass** — disabled, sanity-only (no real search data), or full:
  - *Full bypass* is the main source of per-block loss data for scoring.
    It improves MIP solution quality but requires substantial compute.
  - Ask for granularity: *block* (one complete transformer block) or *subblock*
    (individual attention/FFN sublayers independently).
- **vLLM statistics** — whether to measure actual GPU latency for each
  candidate. Required for hardware-aware latency constraints in MIP.
  Ask for granularity: *block* or *subblock*.
- **Replace-one scoring granularity** — *block* or *subblock*.
- **MIP constraints** — the hardware budget for accepted solutions:
  max parameter count, max latency (e.g. ms/token at a given batch size), or
  max GPU memory. Also ask for: number of top solutions for evaluation, number
  for distillation, and whether to include the teacher as a reference point.
- **Metrics, acceptance gates, and seeds** — which benchmarks define success,
  what the acceptable accuracy gap vs. teacher is, and whether reproducibility
  requires a fixed seed.

### Parallelism

Ask for TP, CP, PP, EP, DP, and sequence parallelism degrees **separately for
smoke and production**. Validate the mesh equation against the declared GPU
count:

```
allocated GPUs = TP × CP × PP × EP × DP
```

Do not assume that named degrees multiply independently — the backend may
require specific combinations (e.g., TP > 1 often requires sequence
parallelism; certain PP schedules require microbatch alignment).

## Emit Runnable Configuration Bundles

After intake, generate two separate namespaces: **smoke** (tiny, fast, full
coverage) and **production** (real scale, pending approval).

Each bundle must contain:
- one canonical experiment YAML accepted by `examples/puzzletron/main.py`;
- referenced topology recipes and data manifests;
- model, tokenizer, processor, dataset, and source revisions;
- a resolved immutable snapshot plus hash;
- artifact root and deterministic stage identities;
- exact direct, Slurm, or SSH launch commands (copy-paste ready);
- exact resume, report-only, and status commands;
- a stage/resource table: stage name, dependencies, topology, expected outputs,
  enabled/optional/disabled status.

Parse both configs with the pipeline loader before presenting them:

```bash
python - <<'PY'
from modelopt.torch.puzzletron.pipeline_config import pipeline_config_from_path
for path in ["smoke.yaml", "production.yaml"]:
    cfg = pipeline_config_from_path(path)
    print(cfg["experiment"]["dir"])
PY
```

## Admit a Pruning Axis

For every candidate axis, verify all of the following before admitting it:

1. A semantic definition: what capacity is removed and which tensors change.
2. A legal discrete value domain with all alignment and grouping constraints.
3. A dynamic slice/mask implementation in Puzzletron.
4. A physical materialization/export implementation.
5. Correct state-dict conversion and load behavior under the admitted axis.
6. Parameter, memory, and runtime accounting.
7. Compatibility with all required backends (HF, AutoModel, vLLM).
8. A physical-versus-dynamic equivalence experiment.

Treat normalization, rotary position encodings, grouped heads, tied weights,
residual projections, recurrent state, and multimodal projectors as coupled
systems. Zeroing output channels is not equivalent to physically reducing a
normalized input.

Reject an axis with an evidence-backed reason when any invariant cannot be met.
Continue with other valid axes unless the whole layer type becomes unprunable.

## Implement and Debug Activation Hooks

An activation hook is a distributed, sample-aware measurement operator. Define:
- exact module boundary and tensor measured;
- reduction dimensions and statistic type;
- valid-token/media mask behavior;
- packed-document boundary handling;
- accumulation dtype and numerical guards;
- shard ownership and durable output schema.

**Distributed rules:**
- DP ranks may process different samples; combine only commutative statistics.
- CP ranks own sequence shards; exclude padding and reduce correctly.
- TP ranks may own features or replicated activations; avoid double counting.
- PP stages observe only local modules; use globally unique layer identities.
- EP ranks observe local experts; retain expert identity and routing metadata.

The manifest must include model/data/config hashes, topology, expected shards,
completed shards, and a completion marker. Resume missing shards only.

## Prove Sorting, Width Ranking, and Slicing

Keep these as three distinct validation checks:

- **Sort sanity:** original vs. sorted vs. reverse-sorted teacher. Outputs and
  loss should match within dtype-aware tolerance. A mismatch is a bug.
- **Width sanity:** does importance ordering improve pruning? Report poor
  rankings as warnings, not hidden failures.
- **Slicing sanity:** for sampled layers and settings, compare all four against
  the original: dynamic sorted, dynamic unsorted, dynamic reverse, and
  physically materialized sorted. Physical is ground truth.

Test at least two representative layers and settings per axis in a production
onboarding. A smoke may use the smallest set that still covers every distinct
axis and layer implementation.

## Handle Multimodal Models

Use processor-native real examples for each selected lane. Preserve modality
ordering, placeholder tokens, media grids/timestamps, cross-attention masks,
labels, and packed boundaries through every pipeline stage.

Determine whether each multimodal tower/projector is prunable, immutable, or
coupled to language width. Include its parameters and memory in accounting even
if it is outside the search space. Report metrics separately per lane and as the
configured aggregate.

## Handle MTP (Multi-Token Prediction)

Inspect MTP depth, shifted-target semantics, head sharing/ties, projector/norm
layers, backend support, and topology behavior. Shifted targets must not cross
padding, packed-document, or modality boundaries.

Report `main_ce`, `mtp_ce`, `main_kd`, and `mtp_kd` separately plus the
weighted total. Do not collapse them into duplicate or ambiguous loss names.

The default smoke includes nonzero MTP CE and KD when the model supports MTP.

## Validate vLLM, Cost, and Memory

Read the relevant implementation before changing formulas:
- `modelopt/torch/puzzletron/export/vllm.py`
- `modelopt/torch/puzzletron/utils/vllm_adapter.py`
- `modelopt/torch/puzzletron/subblock_stats/runtime_vllm.py`
- `modelopt/torch/puzzletron/subblock_stats/calc_subblock_params_and_memory.py`

Count total and active parameters separately. Include embeddings, projectors,
norms, routers, experts + top-k activation, tied weights, MTP, caches,
activations, and server overhead. Make topology assumptions explicit.

## Execute the DAG

Use `modelopt/torch/puzzletron/stages/graph.py` as the authoritative dependency
graph. Schedule independent branches concurrently when resources and artifact
writers do not conflict.

Typical concurrent opportunities: conversion/tokenization + vLLM stats; depth
importance + width importance; vLLM stats + bypass (different GPU allocations).

Only one writer may publish an artifact identity. Aggregate immutable worker
shards into one canonical artifact, then regenerate the cumulative report.

## Configure Bypass Correctly

Bypass replaces a selected layer/subblock's input with the matching teacher
input and computes a local loss for that unit. Key requirements:

- PP: batch size and microbatch scheduling must satisfy the pipeline.
- CP: teacher inputs and masks must share the exact sequence partition.
- TP: enable sequence parallelism when required; avoid duplicated reductions.
- DP: may sample a different valid architecture per rank to obtain multiple
  observations per step.

For every observation, save: step, DP rank, layer/subblock ID, canonical
architecture hash, human-readable config, normalized parameter ratio,
component losses, total loss, seed, and sample identity.

Run two sanity modes before the full bypass: (1) a fixed smallest config that
must clearly overfit, and (2) a diverse resampled mode whose trend should
decrease despite scatter.

## Make Every Expensive Stage Resumable

Expensive stages (bypass, KD, scoring, vLLM stats) must use immutable shards
and transactional checkpoints. A complete training checkpoint stores:
- model and optimizer shards;
- scheduler, scaler, and global step;
- Python, NumPy, CPU/CUDA, sampler, and per-rank RNG states;
- dataloader cursor and exact sample order;
- topology and world-size metadata;
- config/code/model/data identities;
- a manifest and atomic completion marker.

Write to a temporary directory, validate every expected shard, publish
atomically, then update `latest`. Quarantine incomplete transactions.

In the smoke campaign, interrupt and resume at least bypass and Global KD.
Verify no repeated/missing samples, exact global-step continuity, and a valid
cumulative report after resume.

## Launch on Slurm or Bare Metal

**Slurm:** derive commands from the active cluster guide. Capture job IDs and
logs with `tee` + `pipefail`. Checkpoint well before the wall-time limit
(typically 45–55 minutes into a 4-hour slot). Avoid exclusive partial-node
requests.

**Bare metal:**
1. Verify passwordless SSH, identical paths, clocks, ports, and GPU visibility
   on every host.
2. Choose one rendezvous host/port; generate deterministic rank/host mappings.
3. Launch one `torchrun` process group across hosts via SSH.
4. Record remote PIDs, per-rank logs, environment/config hashes.
5. Health-check every rank; terminate all peers on a local failure.
6. Clean up servers and orphan workers explicitly after any failure.

Do not assume a shared Python environment merely because storage is shared.
Verify package and CUDA/driver compatibility on every node independently.

## Full-Coverage Smoke Acceptance

Keep datasets, candidate domains, iterations, and evaluation samples tiny,
but cover:
- every distinct model layer implementation and admitted axis;
- each selected modality lane and fixed/padded/packed layout;
- requested TP/CP/PP/EP/DP paths and required sequence parallelism;
- conversion, tokenization, importance, sorting, dynamic/physical slicing;
- vLLM export/stats and replace-one scoring when applicable;
- MIP, zero-shot evaluation, and a small AIPerf sweep;
- fixed and diverse bypass sanity, plus a short real bypass;
- Global KD sanity and short real KD, including MTP losses when supported;
- checkpoint interruption/resume and cumulative report regeneration.

Record each check and evidence in a smoke manifest. Fail the gate for wrong
equivalence, invalid artifacts, broken resume, backend incompatibility, or
missing report sections.

## Readiness Gate

Before expensive production work, present:
- accepted/rejected axes and source evidence for each;
- smoke outcomes and all unresolved warnings;
- production stage DAG with optional/disabled nodes;
- exact production config paths and their hashes;
- resource/topology estimate per stage and total cost envelope;
- resumability and checkpoint plan;
- exact launch, monitor, and resume commands.

Ask for **one explicit approval**. After approval, execute autonomously within
the authorized cost. Monitor long jobs, diagnose failures from first-rank
evidence, resume durable work, cancel redundant continuations, and never rerun
expensive complete shards.

## Report and Handoff

The HTML report must be cumulative, navigable, and generated from artifact
contracts. It must include:
- experiment ID, pipeline DAG, concise resolved config;
- model/axis inventory, artifact provenance;
- warnings attached to affected table cells;
- all available sanity/results sections;
- partial long-running bypass/KD observations (without erasing earlier data).

Show disabled, pending, and completed nodes; visually distinguish optional
stages. Do not duplicate post-KD metrics in the distillation and post-KD
sections.

At handoff, provide:
- smoke and production config bundle paths/hashes;
- exact run, resume, monitor, report-only, and cleanup commands;
- artifact and HTML report paths;
- completed/pending/disabled DAG nodes;
- measured resource/runtime summary;
- warnings, rejected axes, and recommended next action.

## Stop Conditions

Stop and request direction (do not silently reduce coverage) if:
- checkpoint/data access or license is unresolved;
- the unpruned model cannot run in a required existing backend;
- a required layer type has no valid pruning operation;
- physical and dynamic slicing disagree materially;
- multimodal or MTP target boundaries are incorrect;
- durable resume fails during the smoke;
- production config contains unresolved placeholders;
- requested resources exceed the approved envelope.

## Repository Navigation

Start with `examples/puzzletron/README.md`, `examples/puzzletron/main.py`, and
`modelopt/torch/puzzletron/stages/graph.py`. When present and operating on the
NVIDIA cluster, read `nv-internal/CLUSTER_GUIDE_NV.md` for scheduler behavior.
Read `nv-internal/PUZZLETRON_V2_ENGINEERING_GUIDE.md` for historical review
findings, but validate each claim against current source before acting.

Use symbol search to locate descriptors, hooks, materializers, serializers,
stage implementations, report generators, and tests — paths evolve.

Follow repository coding and test instructions for all code changes. Preserve
dirty work, keep changes generic, and run focused validation proportional to
risk.
