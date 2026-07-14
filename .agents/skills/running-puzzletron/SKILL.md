---
name: running-puzzletron
description: Use when onboarding a new LLM or multimodal model to Puzzletron, running or resuming a Puzzletron pruning campaign, adding model descriptors or pruning axes, validating activation hooks and physical slicing, collecting vLLM statistics, running MIP/evaluation/AIPerf/distillation, or debugging Puzzletron on Slurm or SSH-managed bare metal.
---

# Run Puzzletron

## Objective

Turn a model checkpoint and user requirements into a reproducible, resumable Puzzletron
campaign. Inspect the real implementations before defining pruning axes, prove correctness
with a small but full-coverage smoke campaign, then hand the user runnable production
configuration and commands before spending substantial compute.

Treat the report and durable artifacts as first-class outputs. Never make a stage look
complete merely because a process exited successfully.

## Core Contract

1. Ask the two-phase intake below; do not invent consequential campaign choices.
2. Inspect model source before proposing axes or the second phase of questions.
3. Admit only axes that remain valid through training, physical export, and required
   inference backends.
4. Use physical materialization as the ground truth for dynamic slicing.
5. Run a tiny full-coverage smoke campaign and deliberately test resume.
6. Present one readiness gate before the expensive campaign.
7. Produce concrete smoke and production config bundles plus exact launch commands.
8. Execute the stage DAG, not a hard-coded sequential list; parallelize independent nodes.
9. Regenerate the cumulative report after every completed stage.
10. Preserve or quarantine artifacts. Never silently overwrite incompatible work.

If the user asks only for planning or documentation, stop after producing the requested
files. Do not launch smoke or production work without authorization.

## Phase 1: Establish Identity, Access, and Budget

Ask concise questions, grouped into one message where possible:

- Exact checkpoint URI/path, revision or commit, tokenizer/processor revision, and trust
  policy for remote code.
- Experiment ID, artifact root, and whether existing artifacts may be resumed, imported,
  archived, or must remain untouched.
- Execution environment: Slurm or bare metal; container or host environment; editable
  sibling checkout locations; whether code changes are allowed in Puzzletron, AutoModel,
  and the AnyModel vLLM fork.
- Available node/GPU types and counts, wall-time limits, queue policy, storage capacity,
  deadline, and acceptable smoke/production cost.
- Whether to work in the current checkout or an isolated worktree. Respect dirty user
  changes and repository instructions.
- Required final outcomes: search only, zero-shot evaluation, AIPerf, overfit checks,
  global distillation, post-KD evaluation, or the complete DAG.

For bare metal, also ask for hostnames, slots per host, SSH user, rendezvous/network
interface constraints, and shared-storage root. Require passwordless SSH and a shared
filesystem visible under identical paths on every host. Manage SSH commands, process
IDs, logs, health checks, and cleanup; never copy private keys or weaken host security.

For Slurm, inspect the site guide if one exists. In this repository, read the local
`nv-internal/CLUSTER_GUIDE_NV.md` when present and operating on the NVIDIA cluster;
otherwise discover scheduler policy directly instead of assuming that private guide exists.

## Inspect Before Asking Model-Specific Questions

Resolve the exact model revision locally and inventory every distinct computational layer,
including embeddings, projectors, attention variants, MLP/MoE variants, recurrent/state
space layers, normalization, multimodal towers, routers, language heads, and MTP heads.

For each distinct layer type, read all available implementations in this order:

1. The exact Hugging Face implementation used by the checkpoint.
2. Its vLLM implementation, if the unpruned model is supported.
3. Its AutoModel implementation, if present.
4. Existing Puzzletron descriptors, hooks, materializers, exporters, and cost models.

Trace constructor invariants, tensor shapes, grouped/shared parameters, cache layout,
normalization, residual paths, sharding, collectives, state-dict names, and serialization.
Do not infer a pruning axis from config fields alone.

Create an implementation inventory with:

- layer type and source locations;
- semantic candidate axes and legal value domains;
- coupled dimensions and divisibility/alignment constraints;
- HF, AutoModel, and vLLM constructor/runtime support;
- TP/CP/PP/EP/FSDP implications;
- dynamic hook, physical materialization, export, and cost-accounting status;
- accepted axes, rejected axes, and evidence.

## Select the Model Path

Use an AutoModel descriptor only when AutoModel already natively supports the unpruned
base model. Otherwise implement the model through Puzzletron's HF-native path.

Do not add native AutoModel or vLLM support for an unsupported unpruned model as part of
this workflow. For an already supported model, sibling changes are allowed when needed to
repair broken TP/CP/PP/EP behavior or make an admitted pruned configuration work through
the existing generic AnyModel path.

If the unpruned model runs in vLLM, require every admitted pruned configuration to run in
the repository's AnyModel vLLM fork. Exclude axes that would require a new model-specific
vLLM constructor or backend. For example, Q/K head dimension and V head dimension are
independent mathematically only if all required constructors and kernels actually accept
them independently.

Every layer type must have at least one valid pruning operation or a documented principled
reason it is intentionally immutable. Stop before a costly campaign if a required layer
type has no valid representation.

## Phase 2: Ask Model-Aware Campaign Questions

Present only choices valid for the inspected model.

### Data lanes

Ask independently which supported input combinations to include, such as:

- text only;
- text and image;
- text and audio;
- text and video;
- text with multiple supported media types.

Ask for the real dataset and split for each selected lane, sampling weights, license/access
constraints, maximum samples, ISL/OSL, and evaluation sample counts. Do not replace a
selected multimodal lane with synthetic tensors except for a narrowly scoped unit test.

For each lane, ask independently for one layout:

- **Fixed:** every example is shaped to one known length; simplest and predictable, but
  potentially wasteful and unlike variable production traffic.
- **Padded:** variable examples are batched to a local maximum with attention/label masks;
  conventional and easy to inspect, but includes padding overhead.
- **Packed:** multiple examples share a sequence with explicit document boundaries;
  efficient, but hooks, targets, CP metadata, and media positions must preserve boundaries.

Ask for processor/chat template policy. Require the model's native instruct/chat template
for evaluation and serving. Do not silently substitute generic role tags.

### Search and granularity

Ask independently for:

- accepted candidate ranges for each proven axis;
- maximum depth removals and whether depth units are blocks or subblocks;
- width-importance granularity;
- bypass granularity and whether bypass is disabled, sanity-only, or a search input;
- vLLM-statistics granularity;
- replace-one scoring granularity;
- runtime/memory/parameter constraints and MIP depth options;
- number of best solutions for zero-shot evaluation and AIPerf;
- number of models for distillation sanity and full KD;
- metrics, acceptance gates, seeds, and reproducibility tolerance.

Do not invent a campaign-wide granularity. Each stage owns its own granularity. “Build
Block Library” remains the stage name even when the library records subblock units.

### Parallelism

Ask for allowed TP, CP, PP, EP, DP, sequence parallelism, FSDP/DTensor, batch-size, and
microbatch ranges separately for smoke and production. Validate mesh equations against
the backend rather than assuming every named degree multiplies independently.

## Emit Runnable Configuration Bundles

After intake, generate separate namespaces for smoke and production. Do not leave the
answers as prose.

Each bundle must contain:

- one canonical experiment config accepted by `examples/puzzletron/main.py`;
- referenced topology/worker recipes and data manifests;
- model, tokenizer, processor, dataset, and source revisions;
- a resolved immutable snapshot plus hash;
- artifact root and deterministic stage identities;
- exact direct, Slurm, or SSH launch commands;
- exact resume, report-only, and status commands;
- a stage/resource table showing dependencies, topology, expected outputs, and whether
  the stage is enabled, optional, or deliberately skipped.

The smoke config is executable and tiny. The production config is the actual config the
user can run after approval, not a sketch. Apply smoke-proven fixes to production, resolve
all placeholders, parse both configs with the same loader used by the pipeline, and record
their hashes in the report.

## Admit a Pruning Axis

For every candidate axis, require all of the following:

1. A semantic definition: what capacity is removed and which tensors change.
2. A legal discrete value domain with all alignment and grouping constraints.
3. A dynamic slice/mask implementation.
4. A physical materialization/export implementation.
5. Correct state-dict conversion and load behavior.
6. Parameter, memory, and runtime accounting.
7. Compatibility with all required backends and topology dimensions.
8. A physical-versus-dynamic equivalence experiment.

Treat normalization, rotary position handling, grouped heads, tied weights, residual
projections, recurrent state, and multimodal projectors as coupled systems. Zeroing output
channels is not automatically equivalent to physically reducing a normalized input.

Reject an axis with an evidence-backed reason when any required invariant cannot be met.
Continue with other valid axes unless the whole layer type becomes unprunable.

## Implement and Debug Activation Hooks

An activation hook is a distributed, sample-aware measurement operator. Define:

- exact module boundary and tensor measured;
- statistic and reduction dimensions;
- valid-token/media mask behavior;
- per-sample or aggregate identity;
- packed-document boundary handling;
- accumulation dtype and numerical guards;
- shard ownership and durable output schema.

Verify layouts with text, every selected modality combination, and fixed/padded/packed
batches. Preserve processor outputs, media positions, attention masks, loss masks,
sequence IDs, cumulative lengths, and position IDs through tokenization and collation.

Distributed rules:

- DP ranks may process different samples; combine only commutative statistics and record
  exact sample coverage.
- CP ranks own sequence shards; exclude padding and reduce numerator/count correctly.
- TP ranks may own features or replicated activations; avoid double counting and use SP
  when the backend requires it for TP greater than one.
- PP stages observe only local modules; publish globally unique layer identities.
- EP ranks observe local experts; retain expert identity and account for router/top-k use.
- FSDP/DTensor parameters may be sharded; avoid assuming a full local parameter.

The manifest must include model/data/config hashes, statistic schema, topology, expected
shards, completed shards, sample IDs or ranges, and completion marker. Resume missing
shards only; never trust file presence without identity validation.

## Prove Sorting, Width Ranking, and Slicing

Keep these as distinct checks:

- **Sort sanity:** compare original, sorted, and reverse-sorted teachers without slicing.
  Sorting should preserve outputs/loss within dtype-aware tolerance.
- **Width sanity:** determine whether importance ordering helps pruning; include embedding
  axes when enabled and report poor rankings as warnings, not hidden failures.
- **Slicing sanity:** for sampled layers and settings per axis, compare all results to the
  original teacher: dynamic sorted slice, dynamic unsorted teacher slice, dynamic reverse
  slice, and physically materialized sorted model.

Physical materialization is ground truth. If dynamic and physical results disagree, debug
normalization inputs, residual paths, positional encoding, grouped projections, tied
weights, caches, and masks. Never alter the physical path merely to match a hook.

Test at least two representative layers/settings per axis in a production onboarding,
including one near-teacher width and one more aggressive setting. A smoke may use the
smallest set that still touches every distinct axis and layer implementation.

## Handle Multimodal Models

Use processor-native real examples for each selected lane. Preserve modality ordering,
placeholder tokens, media grids/timestamps, cross-attention masks, labels, and packed
boundaries through conversion, tokenization, importance, bypass, evaluation, and KD.

Determine whether each multimodal tower/projector is prunable, immutable, or coupled to
language width. Include its parameters and memory even if it is outside the search space.
Exercise mixed batches when the processor supports them. Report metrics separately by
lane and as the configured aggregate.

## Handle MTP

Inspect MTP depth, shifted-target semantics, head sharing/ties, projector/norm layers,
backend support, and topology behavior. Couple sorting, pruning, materialization, export,
and parameter accounting wherever the MTP path shares the main model width.

Shifted targets must not cross padding, packed-document, or modality boundaries. Compute
configured CE and KD terms together, using the configured weights. Use the generic
full-vocabulary FlashKLD path for both the language head and MTP heads. Report
`main_ce`, `mtp_ce`, `main_kd`, and `mtp_kd` separately plus the weighted total; do not
collapse them into ambiguous duplicate loss names.

The default full-coverage smoke includes nonzero MTP CE and KD when the model supports
MTP. If it does not, record an explicit not-applicable reason.

## Validate vLLM, Cost, and Memory

Read the relevant implementation before changing formulas:

- `modelopt/torch/puzzletron/export/vllm.py`
- `modelopt/torch/puzzletron/utils/vllm_adapter.py`
- `modelopt/torch/puzzletron/subblock_stats/runtime_vllm.py`
- `modelopt/torch/puzzletron/subblock_stats/calc_subblock_params_and_memory.py`
- `modelopt/torch/puzzletron/benchmarks/aiperf.py`

Count total and active parameters separately. Include embeddings, projectors, norms,
routers, experts and top-k activation, tied weights, MTP, caches, activations, allocator
and graph reservations, kernel workspace, and server overhead. Model attention/MLA,
Mamba/state-space, and GDN/recurrent caches according to their actual implementations.

Make topology assumptions explicit: TP/PP/CP/EP/DP, replicas, dtype, ISL/OSL, batch,
concurrency, and KV/cache blocks. Compare analytical estimates against measured vLLM
statistics and explain residuals before using them as MIP constraints.

For AIPerf, sweep meaningful concurrency and topology settings rather than producing one
point per model. Verify request success and template correctness, not throughput alone.

## Execute the DAG

Treat `modelopt/torch/puzzletron/stages/graph.py` as the authoritative dependency graph.
Do not encode an unrelated linear stage order. Schedule independent branches concurrently
when resources and artifact writers do not conflict.

Typical branches include conversion/tokenization; optional vLLM statistics; depth and
width importance; sorting and sanity checks; optional bypass; library and replace-one
scoring; MIP; zero-shot/AIPerf; and optional KD/post-KD evaluation. Respect the graph in
the current source rather than relying on this summary when they differ.

Only one writer may publish an artifact identity. Aggregate immutable worker shards into
one canonical artifact, then regenerate the cumulative report. Pipeline node state comes
from artifact presence and configuration: `completed`, `pending`, or `disabled`.

## Configure Bypass Correctly

Bypass replaces a selected layer/subblock input with the matching teacher input and
computes the configured local loss for that unit. Align teacher/student microbatches,
packed boundaries, media metadata, PP ownership, and checkpoint identities.

- PP partitions modules; batch size/microbatch scheduling must satisfy the pipeline.
- CP partitions sequence; teacher inputs and masks must share the exact partition.
- TP partitions features; enable sequence parallelism when required and avoid duplicated
  loss reductions.
- EP preserves expert/router ownership and routing metadata.
- DP may sample a different valid architecture per rank to obtain multiple observations
  per step. Optimizer synchronization remains shared unless the selected backend says
  otherwise.

For every observation save step, DP rank, layer/subblock ID, canonical architecture hash,
human-readable config, normalized parameter ratio relative to its teacher unit, component
losses, total loss, seed, and sample identity. Reports should offer a layer/subblock
selector, then an `ALL` or exact-config selector; hovering a point highlights every point
with the same canonical architecture hash.

Run two sanity modes: a fixed smallest configuration that must clearly overfit, and a
diverse resampled mode whose trend should decrease despite scatter. Diagnose targets,
alignment, reductions, learning rate, and optimizer state if they do not.

## Make Every Expensive Stage Resumable

Use immutable shards and transactional checkpoints. A complete training checkpoint stores:

- model and optimizer shards;
- scheduler, scaler, and global step;
- Python, NumPy, CPU/CUDA, sampler, and per-rank architecture RNG states;
- dataloader cursor and exact sample order;
- topology and world-size metadata;
- bypass architecture observations not yet compacted;
- config/code/model/data identities;
- a manifest and atomic completion marker.

Write to a temporary transaction, validate every expected shard, atomically publish it,
then update `latest`. Quarantine incomplete transactions. Permit topology changes only for
artifact types whose schema explicitly supports repartitioning; score/stat shards can
often resume under a new worker topology, while optimizer shards usually cannot.

In the smoke campaign, interrupt and resume at least bypass and Global KD from a new job
or process. Verify no repeated/missing samples, exact global-step continuity, restored RNG
behavior, and a valid cumulative report.

## Launch on Slurm or Bare Metal

For Slurm, derive commands from the active cluster guide and current scheduler state.
Capture job IDs and logs with `tee` plus `pipefail`. Match allocated GPUs to active model
or independent worker meshes. Avoid exclusive partial-node requests and checkpoint well
before the wall-time limit.

For bare metal:

1. Verify passwordless SSH, identical repository/artifact paths, clocks, ports, and GPU
   visibility on every host.
2. Choose one rendezvous host/port and generate deterministic rank/host mappings.
3. Launch one `torchrun` or backend-supported process group across hosts through SSH.
4. Record remote PIDs, per-rank logs, environment/config hashes, and host/GPU assignments.
5. Health-check every rank and terminate all peers on a rank-local failure.
6. Clean up servers and orphan workers explicitly.

Do not assume a shared Python environment merely because storage is shared; verify package
and CUDA/driver compatibility on every node.

## Full-Coverage Smoke Acceptance

Keep datasets, candidate domains, iterations, and evaluation samples tiny, but cover:

- every distinct model layer implementation and admitted axis;
- each selected modality lane and fixed/padded/packed layout;
- requested TP/CP/PP/EP/DP/FSDP paths and required sequence parallelism;
- conversion, tokenization, importance, sorting, dynamic/physical slicing, reporting;
- vLLM export/stats and replace-one scoring when applicable;
- MIP, zero-shot evaluation, and a small AIPerf concurrency sweep;
- fixed and diverse bypass sanity, plus a short real bypass;
- Global KD sanity and short real KD, including MTP losses when supported;
- checkpoint interruption/resume and cumulative report regeneration.

“Full coverage” means semantic/path coverage, not production scale. Record each check and
evidence in a smoke manifest. Fail the gate for wrong equivalence, invalid artifacts,
broken resume, backend incompatibility, or missing report sections. Warnings may remain
only when understood, bounded, visible in the relevant report table, and accepted by the
user.

## Readiness Gate and Production Run

Before expensive work, show:

- accepted/rejected axes and source evidence;
- smoke outcomes and unresolved warnings;
- production stage DAG and optional/disabled nodes;
- exact production config paths and hashes;
- resource/topology estimate per stage and total cost envelope;
- resumability/checkpoint plan;
- exact launch and monitoring commands.

Ask for one explicit approval. After approval, execute the DAG autonomously within the
authorized cost. Monitor long jobs, diagnose failures from first-rank evidence, resume
durable work, cancel redundant continuations, and never rerun expensive complete shards.

## Report and Handoff

The HTML report must be cumulative, experiment-agnostic, navigable, and generated from
artifact contracts rather than ad hoc experiment text. Include the experiment ID, pipeline
DAG, concise resolved config, model/axis inventory, artifact provenance, warnings attached
to affected table cells, all available sanity/results sections, and partial long-running
training observations.

Show disabled, pending, and completed nodes; distinguish optional stages visually. A
disabled stage may still have historical data, but node state follows the active config.
Do not duplicate post-KD metrics in Global KD plots and the post-distillation section.

At handoff provide:

- smoke and production config bundle paths/hashes;
- exact run, resume, monitor, report-only, and cleanup commands;
- artifact and HTML report paths;
- completed/pending/disabled DAG nodes;
- measured resource/runtime summary;
- warnings, rejected axes, and recommended next action.

## Stop Conditions

Stop and request direction before costly work if:

- checkpoint/data access or license is unresolved;
- the unpruned model cannot run in a required existing backend;
- a required layer type has no valid pruning operation;
- physical and dynamic slicing disagree materially;
- multimodal or MTP boundaries are incorrect;
- durable resume fails;
- production config contains unresolved placeholders;
- requested resources exceed the approved envelope.

Do not hide these by reducing coverage, dropping modalities, changing the teacher, or
silently disabling an axis.

## Repository Navigation

Start with `examples/puzzletron/README.md`, `examples/puzzletron/main.py`, and
`modelopt/torch/puzzletron/stages/graph.py`. When present, the ignored NVIDIA-local
`nv-internal/PUZZLETRON_V2_ENGINEERING_GUIDE.md` contains historical review findings;
validate them against current source before acting. Use source search to locate descriptors,
hooks, materializers, serializers, stage implementations, report generators, and tests;
paths evolve, so prefer symbols and artifact contracts over remembered filenames.

Follow repository coding and test instructions for code changes. Preserve dirty work,
make generic fixes, and run focused validation proportional to risk. When only this skill's
documentation is requested, static skill validation is sufficient and campaign execution
remains deferred to the user.
