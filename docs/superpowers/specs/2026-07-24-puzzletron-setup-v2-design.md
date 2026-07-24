# Puzzletron Setup v2 Design

**Date:** 2026-07-24  
**Status:** Approved interaction design; awaiting written-spec review

## Context

The existing setup entry point, `examples/puzzletron/puzzletron_setup.py`, is useful for a
small normal/detailed split but no longer represents the full Puzzletron campaign surface.
In particular:

- `--detailed` controls several unrelated behaviors instead of allowing local decisions.
- Questions are ordered around the current implementation rather than the campaign model.
- A user cannot navigate backward from every prompt.
- Changing an earlier answer either requires restarting or discards too much derived state.
- Most model stages share one hard-coded resource mesh and two special-case meshes.
- Worker counts, parallelism, and batches are not configurable together per stage.
- vLLM statistics describe one workload/topology tuple.
- The MIP editor omits variants, matrices, typed depth, several constraints, and parts of
  homogeneous search.
- The post-MIP editor cannot comfortably review, edit, clone, branch, reorder, or delete nodes.

This design adds a separate, schema-driven wizard while keeping the existing entry point
operational.

## Goals

1. Provide a clean default path that can generate a valid campaign with few decisions.
2. Offer complete guided control over every supported public Puzzletron setup field.
3. Allow backward navigation from every prompt.
4. Preserve and selectively revalidate downstream answers after earlier edits.
5. Configure worker count, parallelism, and batch behavior per applicable stage.
6. Define reusable parallel profiles at first use and reuse or copy them later.
7. Support multiple named vLLM measurement settings.
8. Fully expose the documented MIP run, variant, matrix, objective, constraint, search-space,
   solver, and homogeneous-search capabilities.
9. Provide a usable editor for configurable post-MIP flows.
10. Emit validated smoke and production bundles without launching the orchestrator.

## Non-goals

- Replacing or changing the behavior of `examples/puzzletron/puzzletron_setup.py`.
- Adding a full-screen curses application.
- Providing a raw-YAML escape hatch inside the wizard.
- Exposing private implementation fields that are not part of a supported Puzzletron contract.
- Implementing the reserved PTQ or downstream-evaluation post-MIP node types.
- Launching, monitoring, cancelling, or deleting campaign jobs or artifacts.

## Entry point and package layout

The new public entry point is:

```text
examples/puzzletron/puzzletron_setup_v2.py
```

Its supported command-line arguments are:

```text
--resume PATH
--defaults PATH
```

There is no `--detailed` mode. `--defaults` is explicit; the wizard never discovers or loads a
personal defaults file merely because it exists.

The implementation is divided under `puzzletron_setup/v2/` by responsibility:

- `cli.py`: dependency-light argument handling and user-facing failures.
- `session.py`: navigation, prompt frames, history, and resume position.
- `prompts.py`: visible Back actions and `:back` handling.
- `state.py`: versioned atomic persistence, provenance, and dependency invalidation.
- `defaults.py`: built-in, model-derived, and file-default resolution.
- `resources.py`: reusable model meshes, stage execution cards, packing, and batches.
- `vllm.py`: named vLLM measurement editor.
- `mip.py`: MIP run/variant/matrix editor.
- `post_mip.py`: flow and node editor.
- `validation.py`: cross-section and canonical-schema validation.
- `bundle.py`: v2-state-to-canonical-bundle rendering.
- `wizard.py`: top-level section ordering and orchestration.

Existing lightweight model inspection, model profiles, canonical stage metadata, MIP
normalization, post-MIP compilation, and orchestration compilation are reused instead of
duplicated.

## Interaction model

### Screen structure

Every screen displays:

- Current section and progress.
- The current value.
- The resolved default and its provenance.
- The actions valid at that location.
- A short effective-configuration summary before leaving a section or list item.

Sections and configurable objects use a consistent action vocabulary:

- **Use defaults**
- **Customize**
- **Review**
- **Add**
- **Clone**
- **Edit**
- **Delete**
- **Done**
- **Back**

The default path is always the first action and shows the values that accepting it will use.
The wizard does not ask a global "simple or advanced" question.

### Back navigation

Back is available from every prompt:

- Selection and checkbox prompts contain a visible `← Back` action.
- Text and numeric prompts always display `Type :back to return` and interpret `:back` as
  navigation, not data.

Back returns exactly one prompt frame, including from inside a repeatable editor. Returning
from a nested editor restores its list position and pending item rather than restarting the
section.

### Persistence and resume

Fresh v2 campaigns write a separate versioned `answers_v2.yaml`. Every accepted prompt result,
navigation position, list cursor, reusable profile, and validation status is saved atomically.

Resume restores the exact prompt frame. A defaults file passed during resume participates in
default resolution, but existing explicit answers retain precedence.

### Preserving downstream answers

Each answer field declares dependencies. When an earlier value changes:

1. Downstream answers are retained.
2. Fields that directly or transitively depend on the change are marked stale.
3. Defaults are re-resolved.
4. Stale fields are revalidated.
5. Valid fields are accepted without being asked again.
6. Invalid fields are queued for correction and linked from the review screen.

Changing a display-only name does not invalidate resource or algorithm choices. Changing model
geometry, an axis domain, a referenced vLLM workload, or a resource mesh revalidates only fields
that depend on it.

## Question order

The top-level order is:

1. Campaign directory and optional explicit defaults file.
2. Model source, revision, inspection, and capability summary.
3. Dataset source, modality, layout, and sequence behavior.
4. Runner environment and cluster facts.
5. Pruning and search-space decisions.
6. Pre-MIP stage algorithm, resource, and batch cards.
7. Named vLLM measurements.
8. MIP runs.
9. Post-MIP flows and dynamic-node resource cards.
10. Results location, complete review, validation, and bundle generation.

### Campaign and model entry

After accepting the campaign directory, the wizard proceeds directly to model
selection. It does not show a generic campaign-section
`Use defaults / Customize / Review` menu because the directory is the only
campaign-level input at that point.

The Model prompt is the source chooser itself; it does not first show the
generic per-section `Use defaults / Customize / Review` menu. Its top level is:

1. `Custom`
2. `Nemotron 3`
3. `Qwen 3.5/3.6 Dense`
4. `Qwen 3.5/3.6 MoE`

Selecting a family opens a second prompt containing only that family's
supported checkpoints as concise labels:

- **Nemotron 3**
  - Ultra 550B-A55B
  - Super 120B-A12B
  - Nano 30B-A3B
- **Qwen 3.5/3.6 Dense**
  - Qwen 3.5 0.8B
  - Qwen 3.5 2B
  - Qwen 3.5 4B
  - Qwen 3.5 9B
  - Qwen 3.6 27B
- **Qwen 3.5/3.6 MoE**
  - Qwen 3.6 35B-A3B
  - Qwen 3.5 122B-A10B
  - Qwen 3.5 397B-A17B

Each supported choice resolves to its canonical Hugging Face URL. `Custom`
opens the free-form local-path or Hugging Face model prompt. Back from a family
returns to the top-level Model prompt, and Back from the Model prompt returns
to the preceding wizard boundary. Revision selection and immutable commit
resolution remain unchanged.

Infrastructure precedes stage resource cards so GPU packing, partitions, and derived scheduler
tasks can be shown when a stage is configured. Model inspection precedes infrastructure mesh
validation so MoE and descriptor constraints are known.

## Defaults and provenance

Default precedence, from lowest to highest, is:

1. Built-in wizard defaults.
2. Model- and capability-derived defaults.
3. The explicit `--defaults` file.
4. Preserved answers during resume or backward editing.
5. New explicit user edits.

The review output records both the effective value and its source.

The repository includes `nv-internal/sepehr_defaults.yaml` with this v1 defaults schema:

```yaml
schema_version: 1

data:
  source: /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/engineering/Puzzle-KD-Nemotron-Post-Training-Dataset-v2/

infrastructure:
  execution_contract:
    venv: .venv_new
    container: /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/enroot/pytorch_25p05.sqsh
    container_mounts: /lustre:/lustre
    prerun_commands:
      - source /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/ssameni/setup-envs.sh
  runner:
    kind: slurm
    slurm:
      account: coreai_dlalgo_llm
      partition_cpu: cpu_interactive
```

Unknown defaults keys fail with a path-specific message. Missing keys simply fall through to
the next lower source.

## Stage resource and batch cards

Every enabled compute stage and dynamic post-MIP node receives a resource card. Fields appear
only where the canonical stage or backend supports them.

Applicable cards expose:

- Execution strategy: `single`, `persistent_pool`, or `sharded`.
- Independent model instances/workers.
- TP, CP, PP, DP shard, DP replicate, EP, and sequence parallelism.
- GPU or CPU resource.
- GPUs per node and partition override.
- Stage-specific microbatch, validation microbatch, local batch, global batch, and gradient
  accumulation.
- Stage-specific public algorithm controls such as sample count, sequence length, iterations,
  timeouts, caching, and checkpoint behavior.

The selected strategy is restricted to strategies supported by that stage. The user configures
independent instances/workers; the wizard and orchestrator derive raw scheduler tasks.

### Inline reusable parallel profiles

Profiles are created only when needed. The first model stage offers its resolved mesh. After it
is accepted or customized, it is saved using the stage name by default.

Later stages offer:

```text
Reuse width_importance — TP=1 CP=1 PP=1 DP-shard=2 DP-replicate=2 EP=2 (4 GPUs)
Copy and modify width_importance
Create a new configuration
```

Copying creates an independent profile. Editing a shared profile shows all consumers and
revalidates each one.

### Batch validation

The state records requested and effective batch values separately. For AutoModel model stages,
the minimum scheduling unit includes:

```text
PP × DP shard × DP replicate
```

An incompatible requested micro/local/validation batch is rounded upward to the nearest valid
value. The adjustment is displayed before acceptance and preserved in the review output.
Global/local batch and gradient-accumulation relationships receive their stage-specific
validation rather than being treated as interchangeable fields.

### Derived allocation summary

Before accepting a resource card, the wizard shows:

- Model instances.
- GPUs per model instance.
- Nodes.
- Scheduler tasks.
- GPUs per task.
- Tasks per distributed group.
- Unused allocated GPUs.
- Selected partition.

Invalid topology, task packing, EP divisibility, unsupported sequence parallelism, or capacity
fails immediately. A valid but wasteful allocation requires explicit acceptance.

## Multiple named vLLM measurements

The vLLM section is a repeatable named-settings editor with Add, Clone, Edit, Delete, Review,
and Done.

Each setting is one exact measurement point:

- Unique ID and optional description.
- Block or subblock granularity.
- Measured embedding widths/configurations.
- ISL, OSL, batch size, and maximum concurrent sequences.
- TP, PP, prefill CP, decode CP, executor backend, and GPU group size.
- Warmup iterations, measured iterations, and repeated-block count.
- Cache/merge policy.
- Parameter workers.
- Anchor and trend-validation controls.
- Derived GPU allocation.

MIP workload-dependent constraints select these settings by name. The emitted MIP workload is
derived from the selected measurement, preventing divergent ISL/OSL/batch definitions.

### Runtime configuration extension

The current runtime configuration accepts one ISL/OSL/topology tuple. The implementation adds
a backward-compatible mapping:

```yaml
vllm_stats:
  measurements:
    serving-8k:
      granularity: subblock
      prefill_seq_len: 8192
      generation_seq_len: 1024
      batch_size: 4
      runtime_stats:
        max_num_seqs: 4
        num_warmup_iters: 10
        num_iters: 30
        repeat_block_n_times: 4
        topology: {}
```

The vLLM statistics stage executes, caches, and merges each named measurement independently.
Artifacts include the measurement ID in their identity. Existing single-setting YAML remains
valid by normalizing its legacy fields into one implicit measurement.

The editor shows the number of settings, expected block/subblock configurations, GPUs per
setting, and total work estimate. Duplicate IDs and unsupported topologies fail before bundle
generation.

## MIP editor

The MIP section is a repeatable run editor with Add, Clone, Edit, Delete, Review, and Done.
Each run is divided into separate cards so the primary campaign goal is not conflated with
internal optimizer restrictions.

### Main goal

The main goal defines the primary resource target, for example:

```text
memory <= 75% of teacher at serving-8k
```

It supports every documented friendly metric and, where required, a named vLLM measurement.

### Objectives

One or more quality objectives are selected with explicit minimize/maximize direction. Each
objective produces an independent solver run. Duplicate metric/direction pairs are rejected.

### Internal constraints

Additional constraints are combined with AND. Guided fields cover:

- `params`
- `active_params`
- `memory`
- `runtime`
- `prefill_runtime`
- `throughput`
- `kv_heads`
- `experts`
- A validated `stats.*` metric path

Bounds support directional scalar, minimum, maximum, equality, and range. Values accept
teacher-relative percentages or the documented absolute count, memory, time, and throughput
units. Workload-dependent metrics require a named vLLM measurement.

### Search space

The editor covers:

- All, exact, list, or inclusive-range embedding selection.
- All, exact, list, or inclusive-range total depth.
- Typed subblock depth prefixes where the configured granularity supports them.
- `axes_default` as all or teacher.
- Per-axis all, exact, list, or inclusive-range selectors.

Selections are constrained to inspected and measured domains.

### Solver and homogeneous search

Solver fields include backend, solution count, minimum Hamming distance, and per-solution
timeout.

Homogeneous search includes enablement, keep count/all, objective ranking, or weighted
constraint-closeness ranking. Constraint weights can reference only constraints in the
concrete run.

### Variants and matrices

Runs may contain named variants. The editor shows inherited and overridden fields, permits
editing every supported variant override, and exposes matrix paths supported by the canonical
compiler:

- `embedding`
- `depth`
- `constraints.*`
- `solver.*`
- `homogeneous.*`

The review screen computes and displays the concrete expansion, for example:

```text
2 variants × 3 matrix rows × 2 objectives = 12 independent solves
```

The emitted configuration uses the documented `mip.defaults`, `mip.workloads`, `mip.runs`,
variants, and matrices schema and is normalized by the canonical profile compiler before
bundle publication.

## Post-MIP flow editor

The default is one combined flow per MIP run. Its source may select all or a chosen subset of
that run's variants and objectives. Candidate lineage performs architecture deduplication.

For each flow the user chooses:

- **Use recommended flow**
- **Build flow**

The recommended flow is shown before acceptance and starts without the historical Initial
Filter:

```text
online evaluation
  -> best LM-loss filter
  -> materialize
  -> AIPerf
  -> fastest filter
  -> short global KD
  -> final evaluation
  -> best final-loss filter
```

The template is capability-aware; unavailable operations are omitted with an explanation.

### Building and editing nodes

Adding a node follows this fixed interaction:

1. Select node type.
2. Select candidate input.
3. Select model source: latest, original MIP revision, or an earlier transformer revision.
4. Configure all supported node-specific public fields.
5. Configure stage resources and batches where applicable.
6. Add another node or finish the flow.

The flow editor supports branching, cloning, editing, deleting, and reviewing nodes. Nodes are
displayed in topological order. Reordering is permitted only when dependencies remain valid.
Deleting a referenced node first shows its dependents and requires them to be redirected or
deleted.

Supported node interfaces are:

- `filter`: top-k, threshold, Pareto, and weighted aggregate rank.
- `manual_filter`.
- `materialize`.
- `evaluation`.
- `aiperf`.
- `global_kd`.

PTQ and downstream evaluation are visible as reserved/unavailable capabilities and cannot be
selected.

The editor validates candidate input, model source, artifact kind, metric references,
checkpoint requirements, globally unique node IDs, and DAG cycles immediately. Each dynamic
model node receives the same resource and batch card used by pre-MIP stages.

The final flow review shows:

- Topological text DAG.
- Source run, variants, and objectives.
- Candidate-count limits when statically knowable.
- Config-to-checkpoint artifact transitions.
- Metric dependencies.
- Per-node and aggregate resource estimates.
- Validation errors linked to their owning field.

## Bundle rendering and compatibility

The v2 renderer emits the existing three-file contract:

- `experiment.yaml`
- `runner.yaml`
- `execution.yaml`

for both smoke and production budgets.

The canonical experiment, runner, execution, MIP, post-MIP, and stage compilers remain the
source of truth. V2 answer state is an authoring format, not a new runtime contract. The only
runtime-schema addition is backward-compatible named vLLM measurements.

The existing wizard and existing single-vLLM YAML files continue to work unchanged.

## Validation and failure handling

Before publication, the wizard:

1. Validates defaults-file schema and provenance.
2. Revalidates model and dataset sources.
3. Validates every model mesh, serving topology, EP relation, batch, and allocation.
4. Ensures reusable profile references resolve.
5. Normalizes every MIP run with the canonical profile compiler.
6. Compiles every post-MIP flow with the canonical DAG compiler.
7. Ensures workload, metric, model-source, variant, and objective references resolve.
8. Renders smoke and production bundles.
9. Compiles and dry-runs both bundles.

Validation errors identify the owning section and field. Selecting an error navigates to that
prompt. Correcting it preserves unrelated answers.

Bundle writes are atomic. A failed validation never replaces the last valid generated bundle.
The wizard does not submit an orchestrator job.

## Generated output

A successful run writes:

- `answers_v2.yaml`
- `smoke/experiment.yaml`
- `smoke/runner.yaml`
- `smoke/execution.yaml`
- `production/experiment.yaml`
- `production/runner.yaml`
- `production/execution.yaml`
- A resolved-defaults and provenance summary.
- A campaign README containing exact dry-run and launch commands.

## Test strategy

Focused unit and integration tests cover:

- Back from selection, checkbox, text, numeric, and nested editors.
- Exact resume position.
- Downstream preservation and selective revalidation.
- Defaults precedence, unknown-key errors, and `sepehr_defaults.yaml`.
- Inline resource-profile creation, reuse, copying, and consumer revalidation.
- Model/EP/topology validation.
- Requested/effective batch rounding and global/local batch relationships.
- Derived instances, nodes, scheduler tasks, groups, and unused GPUs.
- Legacy single-vLLM normalization.
- Multiple named vLLM execution identity, caching, and result merging.
- vLLM editor work estimates and MIP workload references.
- Every documented MIP constraint, selector, objective, variant, and matrix form.
- Concrete MIP expansion summaries.
- Post-MIP node add/edit/clone/delete/branch behavior.
- Post-MIP artifact, metric, model-source, and cycle validation.
- Default post-MIP flow without Initial Filter.
- Smoke and production rendering, compilation, and dry-run.
- Coexistence with the existing wizard.

## Success criteria

The feature is complete when:

1. The new entry point generates valid smoke and production bundles without `--detailed`.
2. A user can move backward from every prompt and resume at the exact prior position.
3. Earlier edits preserve every downstream answer that remains valid.
4. Every applicable stage exposes instances, parallelism, and batch controls.
5. Batch adjustments are valid, visible, and recorded as requested/effective values.
6. Later stages can reuse or copy profiles created at first use.
7. Multiple named vLLM settings execute and can be referenced by MIP constraints.
8. The guided MIP editor covers the full documented public schema.
9. The post-MIP editor can build and validate branched flows using all implemented node types.
10. `nv-internal/sepehr_defaults.yaml` supplies the requested personal defaults only when
    explicitly passed.
11. Existing setup entry points and legacy experiment YAML remain compatible.
12. The wizard validates and writes bundles but never launches the orchestrator.
