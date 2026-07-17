---
name: running-puzzletron
description: Use when onboarding a new LLM or multimodal model to Puzzletron, running or resuming a Puzzletron pruning campaign, adding model descriptors or pruning axes, validating activation hooks and physical slicing, collecting vLLM statistics, running MIP/evaluation/AIPerf/distillation, or debugging Puzzletron on Slurm or SSH-managed bare metal.
---

# Run Puzzletron

## Overview

Turn an exact model checkpoint and user requirements into a complete,
reproducible Puzzletron pruning campaign. Inspect actual model/backend source,
admit only physically valid axes, emit hashed smoke and production bundles,
execute the stage DAG, resume durable work, and keep the cumulative report
verified.

If the user asks only for planning or documentation, produce those artifacts
and stop. Do not launch campaign work without explicit authorization.

## Required References

Read each applicable reference completely before acting:

- For model onboarding, descriptors, pruning axes, hooks, sorting, width
  ranking, physical slicing, multimodal models, or MTP, read
  [model-and-axis-validation.md](references/model-and-axis-validation.md).
- For execution environments, AutoModel topology, model-instance semantics,
  the stage DAG, RPC depth evaluation, resumability, Slurm, or bare metal, read
  [distributed-execution.md](references/distributed-execution.md).
- For vLLM candidate construction, exact runtime measurement, cost/memory,
  worker sharding, cache identity, or negative marginals, read
  [vllm-runtime-statistics.md](references/vllm-runtime-statistics.md) and the
  distributed-execution reference.
- For bypass, named MIP profiles, typed depth selection, artifact invalidation,
  reports, trackers, tutorials, or handoff, read
  [bypass-mip-and-reporting.md](references/bypass-mip-and-reporting.md) and the
  distributed-execution reference.
- For a complete campaign, read all four references.

## Core Contract

1. Ask the two-phase intake below; do not invent consequential choices.
2. Inspect exact model and backend source before proposing axes.
3. Admit only axes valid through training, physical export, accounting, and all
   required inference backends.
4. Treat physical materialization as ground truth for dynamic slicing.
5. Run a tiny full-coverage smoke and deliberately test durable resume.
6. Present one explicit readiness gate before expensive production work.
7. Emit runnable smoke and production bundles with immutable identities.
8. Execute the authoritative stage DAG and parallelize independent writers.
9. Regenerate and semantically verify the cumulative report after each stage.
10. Preserve or quarantine incompatible artifacts; never silently overwrite or
    merge them.

## Phase 1: Infrastructure and Goals

Ask one grouped infrastructure message covering:

- exact checkpoint and tokenizer/processor revisions;
- remote-code trust policy and dataset identity/access;
- experiment ID, shared artifact root, and existing-artifact policy;
- Slurm account/partitions/node types/wall limits or bare-metal hosts/slots;
- container image or host Python, mounts, virtual environment, site setup, and
  sibling AutoModel/vLLM/AIPerf checkouts;
- allowed code changes, available GPU/storage budget, deadline, and cost limit;
- required outcome: search, evaluation, AIPerf, bypass, KD, post-KD evaluation,
  or the complete DAG.

When using Slurm, read `nv-internal/CLUSTER_GUIDE_NV.md` if present. For bare
metal, require passwordless SSH, identical shared paths, and an explicit
rendezvous plan.

## Inspect Before Phase 2

Resolve the exact checkpoint revision and inspect every distinct computational
layer: embeddings, attention, dense/MoE experts and routers, recurrent/SSM
layers, norms, multimodal towers/projectors, LM heads, and MTP heads. Read, in
order, the exact Hugging Face implementation, existing vLLM implementation,
AutoModel implementation, and Puzzletron descriptors/hooks/materializers/cost
models.

Build an inventory of semantic axes, legal domains, coupled dimensions,
alignment/grouping constraints, distributed implications, backend support,
dynamic hooks, physical export, and accounting. Record accepted and rejected
axes with evidence.

Use an AutoModel descriptor only when AutoModel already supports the unpruned
model; otherwise use the HF-native path. Do not add a new native AutoModel or
vLLM base model as incidental campaign work. If the base model runs in vLLM,
every admitted pruned configuration must work through the repository's existing
AnyModel vLLM path.

## Phase 2: Model-Aware Questions

Present only options valid for the inspected model. Ask, in one grouped search
message:

- real datasets, splits, modalities, sampling weights, fixed/padded/packed
  layouts, and native chat template;
- legal candidate domains per admitted axis and maximum depth removals;
- bypass, vLLM statistics, and replace-one granularity (`block` or `subblock`);
- parameter, runtime, memory, and active-parameter constraints plus workloads;
- number of solutions per profile for evaluation, AIPerf, and distillation;
- smoke and production TP/CP/PP/EP/DP/sequence-parallel settings;
- evaluation metrics, acceptance gaps, seeds, deadline, and compute envelope.

Explain trade-offs briefly. Ask only follow-ups that materially change the
campaign.

## Emit Runnable Bundles

Create separate `smoke` and `production` namespaces. Each contains:

- one canonical experiment YAML accepted by `examples/puzzletron/main.py`;
- stage-local `automodel.parallel` meshes (never a shared AutoModel recipe YAML), data
  manifests, exact revisions, and execution contract;
- resolved immutable snapshot plus hash and deterministic artifact identities;
- exact launch, resume, monitor, report-only, and safe cleanup commands;
- a stage/resource table with dependencies, topology, outputs, and status.

Parse both configs with the pipeline loader before presenting them. Keep a live
task tracker and a reproduction tutorial from the start; the reporting
reference defines when each may be updated.

## Execute and Validate

Use `modelopt/torch/puzzletron/stages/graph.py` as the dependency authority.
Schedule independent branches concurrently when artifact writers do not
conflict. One writer publishes each canonical identity; immutable worker shards
are aggregated transactionally.

Smoke data and iteration counts may be tiny, but coverage must include every
distinct layer/axis, requested modality/layout, requested distributed path,
conversion, tokenization, importance, sorting, dynamic/physical slicing, vLLM
and scoring when enabled, MIP/evaluation/AIPerf, bypass/KD sanity and short real
runs, interruption/resume, and report regeneration.

Fail readiness for wrong equivalence, broken resume, invalid artifacts, backend
incompatibility, or missing report sections. Warnings about ranking quality may
remain visible without blocking when correctness gates pass and the user has
accepted that policy.

## Readiness Gate

Before production, present:

- accepted/rejected axes and source evidence;
- smoke outcomes and unresolved warnings;
- production DAG with optional/disabled nodes;
- exact config paths and hashes;
- resource/topology estimates and total cost envelope;
- resumability/checkpoint plan;
- launch, monitor, and resume commands.

Ask for one explicit approval. After approval, operate autonomously inside the
authorized cost, monitor long jobs, diagnose from first-rank evidence, resume
durable work, cancel redundant continuations, and never rerun expensive complete
shards.

## Stop Conditions

Stop and request direction rather than silently reducing coverage when:

- checkpoint/data access or licensing is unresolved;
- the unpruned model cannot run in a required existing backend;
- a required layer type has no valid pruning operation;
- physical and dynamic slicing disagree materially;
- multimodal or MTP target boundaries are incorrect;
- smoke resume is not durable;
- production config has unresolved placeholders;
- requested resources exceed the approved envelope.

## Repository Navigation

Start with `examples/puzzletron/README.md`, `examples/puzzletron/main.py`, and
`modelopt/torch/puzzletron/stages/graph.py`. Read the cluster guide when present
and `nv-internal/PUZZLETRON_V2_ENGINEERING_GUIDE.md` for historical findings,
validating each claim against current source. Use symbol search because paths
evolve. Follow repository coding/test instructions, preserve dirty work, keep
changes generic, and validate proportionally to risk.
