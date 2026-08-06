# Puzzletron v2 Support Matrix

This matrix records the exact boundary of the checked-in Puzzletron v2 campaign
evidence. The companion [machine-readable evidence contract](support_evidence.yaml)
keeps the model, configuration relationship, dimensions, settings, stage
statuses, and focused-test gaps synchronized with the reports.

## Evidence policy

**Reported with known slicing warnings** means that a campaign executed the
listed pruning dimensions and settings through a stated stage boundary, while
its report also records unresolved slicing-equivalence warnings. A `completed`
stage status records execution and artifacts; it does not establish that every
correctness gate passed or that the model is supported.

| Evidence level | Meaning |
|---|---|
| Reported with known slicing warnings | A checked-in report shows a named model, dimensions, settings, and completed stage boundary, but unresolved slicing findings prevent a support claim |
| Component-tested | Focused implementation and tests exist, but no checked-in model campaign reports the capability through a stated boundary |
| Experimental backlog | One or more implementation, configuration, test, documentation, or campaign-demonstration gaps remain |

A setup-wizard choice, descriptor, schema entry, or checked-in config does not
establish model support by itself.

## Reported campaign slices

| Model | Executed campaign evidence | Current-code entry and relationship | Reported boundary and warning | Later or mismatched work |
|---|---|---|---|---|
| Nemotron-3 Nano 30B-A3B | [Campaign report](../reports/nemotron3_nano_30b_a3b.html) | [default.yaml](../configs/families/nemotron3/nano_30b_a3b_bf16/runs/default.yaml) is a current-code migration, not a byte-identical copy of the executed production config | Search through zero-shot evaluation, plus global-distillation sanity; the structured report contains 176 slicing-equivalence findings and 2 descriptor-realization-gate findings | AIPerf, full global distillation, and post-distillation evaluation are pending |
| Qwen3.5-9B | [Campaign report](../reports/qwen3p5_9b.html) | [default.yaml](../configs/families/qwen3_5/qwen3p5_9b/runs/default.yaml) reconstructs the current-code campaign; the report used 23 additional overrides | Through AIPerf, global distillation, and post-distillation evaluation; the structured report contains 101 slicing-equivalence findings for deltas above tolerance | The report used embedding widths 4096, 3840, and 3584; the current entry retains only 4096 |

### Nemotron-3 Nano 30B-A3B

- Reported and current embedding widths: 2688, 2560, and 2432.
- Reported model axes: attention KV groups and query heads per group;
  Mamba heads and head dimension; MoE expert count, expert intermediate width,
  shared-expert intermediate width, and top-k; plus conditional depth.
- Reported MIP profiles: `params-075`, `runtime-075`, and `memory-075`,
  covering parameter-count, measured-runtime, and measured-memory goals.
- Completed boundary: conversion, data tokenization, runtime statistics,
  importance, sorting and sanity checks, slicing and bypass sanity, bypass,
  library construction, replacement scoring, MIP, and zero-shot evaluation.
  Global-distillation sanity also completed.
- Known correctness warnings: the structured report contains 176
  slicing-equivalence findings and 2 descriptor-realization-gate findings.
  Resolve them before treating the campaign as model support evidence.
- Not demonstrated as completed: AIPerf, full global distillation, and the
  dependent post-distillation evaluation.

### Qwen3.5-9B

- Reported embedding widths: 4096, 3840, and 3584. The current reconstruction
  config retains only the teacher width, 4096. The report therefore records
  embedding-width pruning, but the checked-in entry does not currently replay
  that part of the search space.
- Reported model axes: FFN intermediate width; attention KV groups and query
  heads per group; Gated DeltaNet key groups, value heads per group, key head
  dimension, and value head dimension; plus conditional depth.
- Reported MIP profile: `latency-095`, using a measured-runtime goal.
- Completed boundary: the full reported path through zero-shot evaluation,
  AIPerf, global distillation, and post-distillation evaluation.
- Known correctness warnings: the structured report contains 101
  slicing-equivalence findings for deltas above the configured tolerance.
  Resolve them before treating the campaign as model support evidence.

## Focused unit-equivalence boundary

The campaign reports mark `slicing_sanity` complete, but both reports also
record unresolved sorted-versus-physical warnings. The completed status is
execution evidence, not proof that the slicing correctness gate passed. It also
does not replace focused executable equivalence tests for every individual
pruning dimension.

The Qwen descriptor and case-construction tests enumerate the available axes,
but the executable dynamic-versus-physical test in
[`test_width_slice_equivalence.py`](../../../tests/unit/torch/puzzletron/test_width_slice_equivalence.py)
currently restricts Qwen to `ffn_intermediate`. Focused executable equivalence
coverage is still needed for Qwen embedding width, attention, Gated DeltaNet,
and conditional-depth dimensions. That test file does not execute a Nemotron
model, so all listed Nemotron dimensions still need campaign-specific focused
unit-equivalence coverage.

These test gaps are separate from the report warnings above. Both remain
visible so a completed stage is not mistaken for model support or complete
per-axis regression coverage.

## Experimental backlog

The items below are intentionally not support claims.

| Candidate capability | Evidence already present | Evidence still required for promotion |
|---|---|---|
| Qwen3.6-35B-A3B MoE campaign | [Pinned model definition](../configs/families/qwen3_5/qwen3p6_35b_a3b/model.yaml), [production config](../configs/families/qwen3_5/qwen3p6_35b_a3b/runs/production.yaml), descriptor/config tests, and orchestration config coverage | A checked smoke or end-to-end model campaign that states its dimensions, settings, and completed boundary, plus a verified report and operating notes |
| VLM and multimodal campaigns | Dataset materialization, media-aware batching, descriptor paths, and focused tests for multimodal data, forwarding, and global KD | A pinned VLM config, model-level axis and materialization evidence, backend execution, end-to-end evaluation, a verified report, and a documented example |
| Unified online-MIP evaluation and topology-matrix AIPerf orchestration | Standalone profile tools, campaign configs, and focused orchestration/reporting tests | Integration into the canonical campaign DAG plus a checked end-to-end report produced through that unified path |
| Additional model, axis, input-layout, and parallel-topology combinations | Reusable descriptors, schema entries, and component tests for several combinations | Per-combination semantic validation, physical materialization and reload, backend execution, dynamic-versus-physical equivalence, and campaign evidence through a stated boundary |

## Promotion checklist

Before moving an item out of the backlog:

1. Pin the executed model revision, config, overrides, pruning dimensions, and
   material settings.
2. Test axis semantics, legal values, dynamic execution, physical
   materialization, reload, accounting, backend execution, and
   dynamic-versus-physical equivalence. List any remaining focused-test gaps.
3. Run a model campaign through an explicitly named boundary and exercise every
   dimension included in the claim.
4. Check in a report with stage statuses, sanity checks, and
   teacher-versus-candidate evaluation. List enabled stages that remain pending
   or failed.
5. Add operating notes and update both this matrix and
   `support_evidence.yaml` with the same evidence.
