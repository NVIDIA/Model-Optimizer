# Puzzletron AIPerf and Parallelism Validation Design

## Problem

Puzzletron setup currently presents expert parallelism as an independent numeric
vLLM setting and carries that value through resource sizing and AIPerf launch
configuration. vLLM does not expose an independent expert-parallel size. For an
MoE model, enabling vLLM expert parallelism makes the effective expert-parallel
size equal to tensor parallelism multiplied by data parallelism.

The setup wizard also validates only a small subset of AutoModel parallelism
constraints. A profile can therefore be accepted even when it is incompatible
with the teacher geometry or with one of the model geometries selected by the
user's pruning axes. Reused profiles, defaults, and hand-edited persisted state
can bypass the existing checks.

Invalid settings must be rejected explicitly. The wizard must explain the
conflict and ask the user to choose another setting rather than silently
rewriting the requested degree.

## Goals

1. Ask AIPerf workload questions using AIPerf terminology and map each answer to
   the correct AIPerf command-line option.
2. Ask vLLM serving topology questions separately from AIPerf workload
   questions.
3. Represent vLLM expert parallelism as an enable/disable choice, not an
   independent numeric degree.
4. Explain in the MoE serving prompt that effective expert parallelism is
   `TP * DP` when expert parallelism is enabled.
5. Reject stage parallelism that is incompatible with any model geometry that
   the stage can encounter.
6. Apply the same rules to newly created, reused, default, and persisted
   profiles.
7. Keep the changes localized and generic across supported dense, MoE, GDN, and
   Mamba-style descriptors.

## Non-goals

- Automatically changing an invalid degree to a nearby valid value.
- Requiring pipeline stages to receive equal layer counts. Uneven pipeline
  partitioning is supported, so PP is not validated against layer-count
  divisibility.
- Validating pruning-axis combinations themselves; this design validates their
  compatibility with execution parallelism.
- Exposing every AIPerf command-line option in the setup wizard.

## Proposed Architecture

Add one stage-aware compatibility component under `puzzletron_setup/v2/`.
It will consume:

- the discovered `ModelInventory`;
- the selected pruning-axis values;
- a stage identifier;
- an AutoModel parallel profile or vLLM serving topology; and
- stage parameters such as sequence length.

It will return structured compatibility issues. Each issue identifies the
stage, setting, conflicting geometry values, and acceptable alternatives where
they can be computed concisely.

The component will be used at three boundaries:

1. The interactive wizard, to reject a selection and return to the relevant
   prompt.
2. Final state validation, to catch reused, default, or hand-edited persisted
   profiles.
3. Bundle rendering, as a defensive last boundary before invalid commands are
   emitted.

The compatibility logic will remain independent from prompt rendering so it can
be tested with small literal inventories.

## AIPerf Workload and vLLM Serving Questions

The setup UI will distinguish workload generation from server topology.

### AIPerf workload

| Wizard answer | AIPerf command line |
| --- | --- |
| Input token count | `--synthetic-input-tokens-mean` |
| Fixed input length | `--synthetic-input-tokens-stddev 0` |
| Output token count | `--output-tokens-mean` |
| Fixed output length | `--output-tokens-stddev 0` |
| Concurrency | `--concurrency` |
| Request count | `--request-count` |

The existing `minimum_request_count` and `requests_per_concurrency` smoke
settings will be consumed while deriving request counts. They will not be
forwarded as unsupported arguments to the AIPerf runner.

### vLLM topology

The wizard will ask for:

- tensor parallel size (TP);
- pipeline parallel size (PP);
- data parallel size (DP);
- prefill context parallel size;
- decode context parallel size; and
- for an MoE model, whether to enable expert parallelism.

The MoE prompt will state:

> vLLM does not accept a separate EP size. When expert parallelism is enabled,
> effective EP is TP * DP.

The vLLM command will receive `--enable-expert-parallel` only when selected.
There will be no generated numeric expert-parallel argument.

The serving allocation size is:

`TP * PP * DP * prefill_context_parallel_size`

Decode context parallelism reuses tensor-parallel workers and does not multiply
the GPU allocation.

For compatibility with persisted configurations, a legacy numeric
`expert_parallel_size` is accepted only when it unambiguously represents the
new semantics:

- `1` means expert parallelism is disabled; or
- `TP * DP` means expert parallelism is enabled.

Any other numeric value is rejected with a migration message.

## Geometry Domains

Validation will operate on sets of geometry values rather than only the teacher
value.

The teacher domain contains the original model dimensions discovered in the
inventory. A candidate domain contains the teacher value plus every value
selected for the corresponding pruning axis. Derived dimensions are computed
for each relevant combination, including:

- query heads from KV groups and query heads per group;
- GDN value heads from key groups and value heads per group; and
- effective vLLM EP from TP and DP when expert parallelism is enabled.

This makes validation conservative: a stage profile is accepted only if every
geometry that the stage may load is compatible.

## Stage Geometry Scope

### Teacher-only AutoModel stages

These stages load the teacher's physical tensor shapes:

- depth importance;
- width importance;
- sort sanity;
- bypass;
- bypass sanity; and
- other equivalent full-teacher profiling stages.

Bypass stages apply masking and therefore retain teacher tensor geometry. They
do not need to be compatible with physically pruned candidate sizes.

### Candidate-aware AutoModel stages

These stages can load or materialize selected candidate geometries:

- width sanity;
- slicing sanity;
- replacement scoring;
- post-MIP evaluation; and
- global knowledge distillation.

Their profiles must be compatible with the teacher and all selected values for
every axis they can encounter.

### Candidate-aware vLLM stages

These stages serve materialized candidate models:

- vLLM statistics collection; and
- post-MIP AIPerf.

Their vLLM topology must be compatible with every selected candidate geometry.

Single-GPU bookkeeping, sorting, and library-building stages that do not execute
a distributed model are outside this validation.

## AutoModel Compatibility Rules

For each geometry in the applicable stage domain:

- TP must divide hidden widths that are tensor-sharded.
- TP must divide query-head and KV-head counts required by the native
  AutoModel attention implementation.
- TP must divide dense FFN, expert FFN, shared-expert FFN, and latent widths
  that are tensor-sharded.
- TP must divide applicable GDN key/value head counts and Mamba head counts.
- EP must divide every applicable expert count.
- Dense models require EP of one.
- The existing DP-shard/EP relationship remains enforced.
- Sequence parallelism requires TP greater than one.
- Context parallelism must be compatible with the configured sequence length.

PP must be positive, but it is not constrained by even layer partitioning or by
layer-count divisibility.

## vLLM Compatibility Rules

For each candidate geometry:

- TP must divide the query-head count.
- Decode context parallelism must satisfy vLLM's TP/DCP relationship.
- The detailed query-head/KV-head constraints required by non-MLA decode
  context parallelism must hold.
- Expert parallelism can be enabled only for an MoE model.
- When enabled, effective EP is `TP * DP` and must be compatible with every
  selected expert count supported by the serving backend.
- The requested topology must fit the computed GPU allocation.

PP remains unconstrained by equal layer distribution.

## Interactive Rejection

Profile creation, profile reuse, and serving topology selection will all run the
same compatibility check before acceptance. On failure, the wizard will print
all relevant conflicts and return to the profile or topology prompt.

Example:

```text
post.params.serving: TP=8 is incompatible with query-head counts [24, 30].
Choose a TP that divides every encountered value; valid choices include
[1, 2, 3, 6].
```

The wizard will never substitute a value automatically. If a built-in default
is incompatible, the user will be asked to customize the setting.

Final state validation will repeat these checks so persisted or manually edited
state cannot bypass the interactive validation.

## Testing Strategy

Implementation will proceed test-first with:

1. Table-driven unit tests for teacher and candidate geometry domains.
2. AutoModel constraint tests for dense, MoE, GDN, and Mamba inventories.
3. Stage-scope tests proving bypass uses only teacher geometry while
   width/slicing and post-MIP stages include selected candidate sizes.
4. vLLM tests for TP, DP, DCP, and effective `EP = TP * DP`.
5. Wizard-session tests proving invalid created and reused profiles are rejected
   and reprompted.
6. Persisted-state and bundle-boundary rejection tests.
7. Exact AIPerf workload and vLLM server command tests.
8. Legacy numeric EP migration and rejection tests.
9. Smoke and production bundle rendering tests.
10. Focused orchestration checks followed by an 8-GPU interactive-node serving
    test against a representative subset of saved models.

## Compatibility and Change Scope

The implementation will preserve existing dense and Qwen behavior by deriving
rules from model inventory rather than model names. Existing configuration keys
will be read through the explicit legacy EP compatibility rule, while newly
rendered configurations use the boolean expert-parallel setting.

Changes will be concentrated in the v2 setup validation and prompting paths,
the shared AIPerf/vLLM topology adapter, and focused tests. Unrelated
orchestration and pruning behavior will not be restructured.
