# Bypass, MIP, and Reporting

## Configure Bypass

Bypass replaces a selected block/subblock input with the matching teacher input
and computes a local loss. Ensure PP batch/microbatch compatibility, identical
CP partitions and masks, required TP sequence parallelism without duplicate
reductions, and valid per-DP architecture sampling.

Every observation records step, DP rank, layer/subblock, canonical architecture
hash, human-readable config, normalized parameter ratio, component/total loss,
seed, and sample identity.

Run two sanity modes before production:

1. a fixed smallest config that clearly overfits;
2. a diverse resampled schedule whose trend decreases despite scatter.

Define production scale with total tokens, sequence length, tokens/update,
microbatch, gradient accumulation, and derived target steps. On a fresh run,
initialize the current runner's state cursors at `step_num=1`, `iter_num=0`, and
`token_count=0`; these are resume counters, not target totals.

Overfit success validates boundaries and optimization mechanics, not production
LR. Before spending the full budget, require an early finite production trend
with improvement represented across every sampled width/config family. Record
configured maximum LR, observed LR, scheduler phase, loss windows, and gradient
norms. A successful exit with flat/worsening loss is rejected evidence.

Measure tokens/second before adding nodes. Communication, restore, and routing
overhead may make a larger DP topology slower. Keep the smallest topology with
the best verified throughput that fits memory. When monitoring a resume, join
the durable checkpoint/history with newer live-log records across the boundary.

## Compile Named MIP Profiles

Read `examples/puzzletron/docs/mip_profiles.md`. Define named workloads in the
root MIP namespace and reference them from runtime/memory constraints. Multiple
constraints may target different workloads; resolve each percentage against
that workload's teacher total. Missing workload data is invalid, not a reason to
relax the constraint.

Support exact values, percentages, ranges, and lists. Scalar shorthand is
equality; use explicit `min`/`max` for inequalities. Use
`num_homogeneous_solutions` for homogeneous Cartesian baselines (`-1` means all)
and restricted-axis domains to compare mix-and-match against single-axis
searches. Omitted domains inherit all legal values; explicitly fixed domains
stay at the teacher value.

Depth collection remains one global iterative trajectory. Profiles may request
a total prefix or, only with subblock depth importance, typed attention,
Mamba/SSM, MoE, or dense-FFN counts. Expand lists by Cartesian product, select
each kind's prefix from the global trajectory, and union while preserving global
order. Omitted kinds mean zero. Never mix `total` and typed counts. Validate
availability before solving.

Identity scenarios with the complete typed depth map, profile, width, domains,
constraints, and workload—not only total removals. Equal-total typed selections
remain distinct through resume, materialization, reports, evaluation, AIPerf,
and KD. Keep profile identity even when two profiles produce the same structure.

## Artifact Invalidation

Changing LR, token budget, estimator schema, physical candidate construction,
workload, or execution contract creates a new semantic identity. Quarantine
incompatible canonical outputs and selected downstream descendants. Never merge
old/new shards because filenames or shapes match.

Before rerunning an invalidated stage, regenerate the report and verify old
results disappeared and the stage is pending. Rebuild canonical aggregates only
from compatible completed shards. Preserve rejected evidence under a clear
non-canonical namespace.

## Report Verification

The cumulative HTML is generated from canonical artifact contracts after each
completed stage and may show partial long-running observations without marking
them complete. Include experiment identity, DAG, resolved config, model/axis
inventory, provenance, cell-level warnings, sanity/results, and pending,
disabled, optional, and completed nodes. Do not duplicate post-KD metrics.

Use model identity precedence: configured display name, declared Hugging Face
repository plus revision, model metadata, then local path. Never title the
report with a resolved HF cache snapshot when repository identity exists.

Verification is semantic: parse the report and assert model identity, stage
state, candidate/width/depth counts, profile names, warnings, and canonical
paths. Partial caches and worker shards are progress only. Report-only fixes do
not invalidate expensive compatible compute.

## Tracker, Tutorial, and Handoff

Maintain a live task tracker containing jobs, progress, evidence, dependencies,
and next actions. It may record partial observations. Maintain a reproduction
tutorial containing only commands and conclusions whose artifacts and report
sections have been verified.

At handoff provide bundle paths/hashes, run/resume/monitor/report/cleanup
commands, artifact/report paths, DAG states, measured resources/runtime,
warnings, rejected axes, and the recommended next action.

