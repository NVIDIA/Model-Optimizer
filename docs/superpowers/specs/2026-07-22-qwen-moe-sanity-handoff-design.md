# Qwen-MoE Puzzletron Sanity Handoff Design

## Goal

Prepare the existing Qwen-MoE campaign for a user-driven `bypass_sanity` resume while keeping all behavioral fixes generic to Puzzletron and AnyModel.

## Runtime-stat finalization

The packed vLLM launcher currently stops after its shard workers finish, unlike the normal `vllm_stats` stage, which also validates the aggregate and generates the offline report sidecars. Extract a shared finalizer that validates `subblock_stats.json` and writes `artifacts/vllm_stats/summary.json` plus its HTML, CSV, and warning outputs. The normal stage and the packed launcher will call the same finalizer. Only the packed group containing logical shard zero may finalize, after all of its workers have exited successfully; shard zero already waits for the complete shared aggregate.

The existing valid Qwen-MoE aggregate will be finalized without recollecting GPU statistics.

## Sanity warning policy

Add one root-level configuration option:

```yaml
sanity:
  fail_on_warnings: false
```

All stages that use `complete_sanity_stage` will retain their factual `passed`, `verdict`, and `findings` outputs. With `fail_on_warnings: false`, warning-only verdicts complete successfully and orchestration may continue. With `true`, the same evidence is written but the stage result and manifest are failed, causing the worker and dependent orchestration branch to stop. Errors raised before verdict construction remain failures regardless of this option.

The `sanity` section participates in stage semantic identity so changing policy invalidates stale completion manifests consistently.

## Qwen-MoE campaign configuration

The completed depth trajectory contains removals zero through two. Align downstream MIP configuration with it by setting `depth_scenario_count: 3` and the shared profile depth range to `[0, 2]`.

Set both sorted and reverse-sorted LM-loss tolerances to `0.015` in the Qwen-MoE production configuration. This accepts the already measured reverse delta of approximately `0.00515` while keeping the threshold below the model descriptor's established equivalence tolerance. Regenerate the sort summary from the existing measurements.

Set `sanity.fail_on_warnings: false` for this campaign so warning-only sort, slicing, bypass, and later sanity stages remain visible but non-blocking.

## Slicing replay

Re-run the failed GDN key-head-dimension slicing evidence against the existing physical realizations using the generic logical-to-physical scale correction. Compare the new logical/physical deltas with the quarantined baseline. If they improve and meet configured equivalence tolerances, publish canonical width/slicing summaries. If they do not improve, stop further speculative slicing changes and report the result.

## Cleanup, report, and handoff

Preserve durable rankings, trajectories, runtime statistics, manifests, and diagnostic evidence. Remove only confirmed temporary physical checkpoints, stale orchestration handles/snapshots, and superseded partial outputs after canonical artifacts are verified. Regenerate the campaign report and provide the exact orchestrator command that begins at `bypass_sanity`; do not run bypass training.

## Verification

Use focused CPU tests for the shared warning policy, semantic configuration, and vLLM packed finalization. Run the affected Puzzletron unit tests and formatting checks. GPU validation is limited to the focused slicing replay on a held interactive Slurm node. Validate final JSON artifacts and campaign report before handoff.
