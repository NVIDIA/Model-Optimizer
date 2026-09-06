# Qwen 3.5 0.8B VLM pruning smoke

This guide provides two runnable end-to-end examples for Qwen 3.5 0.8B:

- The [all-axis reproducibility smoke](#run-the-reproducibility-smoke) prepares
  pinned data, exercises every reducible axis, materializes and reloads three
  students, runs two KD steps plus teacher/pre-KD/post-KD evaluation and a small
  serving cell, verifies stable result fields, and resumes with the same command.
- The [quick quality comparison](#run-a-quick-quality-comparison) prepares
  its pinned data, screens the all-axis search globally, materializes two
  candidates, gives both the same 64-step KD exposure, and evaluates and serves
  both before its configured rule retains one.

Both routes use deterministic evaluation rows and produce a cumulative report.
Their small evaluation sets and serving workloads test the pipeline; use larger
benchmark and serving workloads to measure model quality or throughput.
`full_vlm_smoke.yaml` provides a portable FFN-only smoke.

## Before you start

Prepare the setup and worker environments described in
[environment setup](environment_setup.md). The worker environment needs
ModelOpt, NeMo AutoModel's Qwen 3.5 VLM support, the Puzzletron requirements,
and the AIPerf/vLLM runtime selected by your runner.

Workers also need access to:

- `Qwen/Qwen3.5-0.8B` at the revision in
  `configs/families/qwen3_5/qwen3p5_0p8b/model.yaml`;
- a writable `HF_HOME` for the pinned evaluation snapshots and prepared media;
- a shared writable location for the campaign output and prepared dataset.

The `prepare_dataset` stage downloads and validates the evaluation tasks selected
by the run config in the runner's `HF_HOME`, including required video media.
Snapshot and media inventories make resume reject partial data. See
[cache benchmark data](vlm_checkpoint_evaluation.md#cache-benchmark-data) for
standalone preparation and recovery behavior. If workers cannot access the
network, populate those exact caches before launch and mount them through the
runner. Keep the configured model identity instead of replacing it with a
machine-specific path.

## Prepare the output root

Choose a path visible to every worker. The run config pins the dataset revision
to an immutable Hugging Face commit and materializes the dataset below this
root as the first resumable CPU stage.

```bash
export PUZZLETRON_RUN_ROOT=/path/to/qwen3p5_0p8b_vlm_smoke
export HF_HOME=/path/visible-at-the-same-location-to-controller-and-workers/hf-home
```

Export `HF_HOME` in the controller shell before invoking the orchestrator;
Hydra seals that exact worker-visible path into the preparation stage's resume
identity. A matching acquisition manifest is reused on resume; a mismatched or
unsealed destination is rejected instead of silently overwritten.

## Configure the runner

The example provides the experiment and execution settings. Copy the runner
template into the output directory and replace its site-specific placeholders:

```bash
RUNNER_TEMPLATE=examples/puzzletron/configs/orchestration/qwen3p5_0p8b/runner.slurm.yaml
RUNNER="$PUZZLETRON_RUN_ROOT/runner.slurm.yaml"

mkdir -p "$PUZZLETRON_RUN_ROOT"
if test -e "$RUNNER"; then
  echo "runner already exists: $RUNNER" >&2
  exit 1
fi
cp "$RUNNER_TEMPLATE" "$RUNNER"
${EDITOR:-vi} "$RUNNER"

if rg -n 'REPLACE_WITH_' "$RUNNER"; then
  echo "replace every runner placeholder before continuing" >&2
  exit 1
fi
```

Set the repository, environment, container, mounts, Slurm account, and
partition for your site. The model cache, benchmark cache, and output directory
must be visible inside the worker environment.

## Run the reproducibility smoke

Use `vlm_reproducibility_smoke.yaml` for the unattended reference smoke. It
uses an immutable 24-row evaluation profile: eight RealWorldQA rows, eight
MMMU validation rows, and eight `action_sequence` MVBench rows. The MIP inputs
include separate query-head and KV-group variants plus one combined sentinel.
The combined sentinel binds FFN width to layer 1, grouped attention to layer
19, and GDN geometry to layer 0; other layers keep teacher geometry.

One command prepares or validates the pinned training and evaluation data,
then runs or resumes materialization, checkpoint reload, teacher/pre-KD
evaluation, two-step KD, post-KD evaluation, the serving mechanics cell, the
final report, and the checked expectation contract:

```bash
EXPERIMENT=examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/vlm_reproducibility_smoke.yaml
EXPECT=examples/puzzletron/expected/qwen3p5_0p8b_vlm_smoke_v1.json
EXECUTION=examples/puzzletron/configs/orchestration/qwen3p5_0p8b/execution.vlm_reproducibility_smoke.yaml

python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full \
  --expect "$EXPECT"
```

Rerun the same command against the same output root to resume. Exit status 0
means the campaign and expectation comparison passed, 1 means comparable
results regressed, and 2 means required result fields were missing, incomplete,
invalid, or nonfinite. The comparison checks architecture, dataset, evaluator,
geometry, parameter counts, tensor counts, KD exposure, and stage completion.
It records checkpoint contents, evaluator source revision, scores, and serving
timing without requiring those values to match exactly.

## Run a quick quality comparison

The `e2e_vlm_quality_comparison.yaml` route is a compact end-to-end example. The
grid has eight feasible width/depth scenarios; with
`num_solutions: 8` per scenario it can emit up to 64 solutions. The route scores
every solution on the same 16 image-text samples, globally retains two, and
gives both candidates the same 64-step KD exposure with global batch size four.
It then compares the students with the pinned teacher on deterministic
RealWorldQA, MMMU, and MVBench rows. Evaluations are resumable and are reused
only when checkpoint identity and evaluator artifacts still match. Results
include student and teacher metrics and their deltas. This route does not
enforce a minimum score.

Use `e2e_vlm_quality_comparison_extended.yaml` to run the comparison after the
extended smoke candidate.

Run the comparison route with a site-specific runner and a distinct output
root. Its preparation stage fills or validates the pinned RealWorldQA, MMMU,
and MVBench snapshots in a writable runner `HF_HOME`; offline runners must
prepopulate that cache. The route requires a GPU and is not part of default CI:

```bash
EXPERIMENT=examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/e2e_vlm_quality_comparison.yaml
EXECUTION=examples/puzzletron/configs/orchestration/qwen3p5_0p8b/execution.e2e_vlm_quality_comparison.yaml
RUNNER=/path/to/site-specific/runner.slurm.yaml
export PUZZLETRON_RUN_ROOT=/path/to/qwen3p5_0p8b_e2e_vlm_quality_comparison

python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full --dry-run
```

Inspect the compiled stage order and one-GPU-per-worker allocation before
launch. The two candidate workers may run concurrently. Then launch or resume
the exact same three-input campaign with:

```bash
python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full
```

## Run the FFN-only smoke

This route searches two FFN sizes, materializes and reloads two candidates,
selects one using its small serving cell, and runs two KD steps. Use the same
runner and output-root setup with these files:

```bash
EXPERIMENT=examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/full_vlm_smoke.yaml
EXECUTION=examples/puzzletron/configs/orchestration/qwen3p5_0p8b/execution.full_vlm_smoke.yaml

python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full --dry-run
```

Inspect the plan, then omit `--dry-run` to launch or resume it. Its synthetic
serving requests check mechanics only; use a representative workload for
throughput measurements.

## Understand the broader campaign example

`vlm_campaign.yaml` is an illustrative campaign definition. It is not a
recommended KD recipe, convergence schedule, or default candidate-selection
policy. It searches hidden width,
heterogeneous FFN width, depth, grouped-attention geometry, and GDN key groups
and head dimensions. It uses the `params-90` MIP profile and an image-text
LM-loss screen. It requests up to eight
heterogeneous MIP solutions for each feasible width/depth scenario and retains
four candidates for matched KD. Candidate generation,
physical materialization, reload validation, and serving all fail closed when a
requested geometry is unsupported. The `3328` and `3072` FFN widths are
separate controls.

Every LM-loss-retained candidate follows one resumable trajectory from the same
immutable pre-KD materialized checkpoint. The 64, 128, and 256 values are
example observation points on that single trajectory, not three independent KD
runs. They demonstrate resume and evaluation at increasing exposure; they do
not establish an appropriate training duration for another campaign. The
configured rule retains one candidate after all three checkpoints have been
evaluated, using equal-rank aggregation of the 256-step RealWorldQA and MMMU
scores. The
placement-bound all-axis sentinel follows an independent trajectory, so it
cannot be removed by the LM-loss top-k screen.

The `3328` and `3072` controls follow their own otherwise identical 64/128/256
learning curves. Every milestone uses the same evaluator
`qwen35_vlm_realworldqa64_mmmu120_mvbench160_frozen_rows_v3` profile and its
fixed rows: 64 RealWorldQA rows, 120 MMMU validation rows, and 160 MVBench rows.
The teacher result is computed once and reused for every control, candidate,
and milestone. Each KD record reports batch size, cumulative examples,
effective tokens, a padded-token upper bound, and GPU time. The learning-curve
manifest records the pre-KD checkpoint, each KD checkpoint, metrics, training
exposure, teacher identity, and selected evaluation rows.

Use the `qwen35-vlm-judge-free8-all-rows-v1` profile separately for an
eight-task, all-rows evaluation.

Grouped-attention and GDN reductions use a limited native runtime.
Compact execution supports only unsharded SDPA `Qwen3NextAttention` and the
single-device, context-parallel-one, non-packed GDN path. Transformer Engine,
DTensor or tensor-parallel attention, context parallelism greater than one,
and packed GDN execution fail closed. Validate isolated candidate execution and
physical materialization before relying on a combined search result.
`gdn_value_heads_per_group` remains disabled because its teacher value is
already one.

## Run the all-axis campaign

Dry-run `vlm_campaign.yaml` with the site-specific runner and a distinct output
root before launch.

The route evaluates four ranked candidates and applies its aggregate-rank rule
at 256 steps. It runs matched 64/128/256-step example trajectories for those
candidates, the placement-bound all-axis sentinel, and the two FFN controls.
Candidate and sentinel checkpoints each receive a
three-repetition AIPerf cell; every repetition uses 32 warmup requests followed
by 64 measured requests.

```bash
EXPERIMENT=examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/vlm_campaign.yaml
EXECUTION=examples/puzzletron/configs/orchestration/qwen3p5_0p8b/execution.vlm_admitted_axes_campaign.yaml
RUNNER=/path/to/site-specific/runner.slurm.yaml
export PUZZLETRON_RUN_ROOT=/path/to/qwen3p5_0p8b_vlm_campaign

python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full --dry-run
```

Inspect the compiled stage order, input identities, supported runtime modes, and
one-GPU-per-worker allocation; up to four candidate workers may run
concurrently. Then omit `--dry-run`; rerunning that command resumes the
campaign. This campaign does not include an expected-result baseline. Change
or remove the KD milestones and selection rule when adapting the example to a
different experimental question.

## Evaluate a saved checkpoint separately

The campaign records checkpoint evaluations with each candidate. For
standalone preflight and evaluation commands, suites, result files, and cache
preparation, use [VLM checkpoint evaluation](vlm_checkpoint_evaluation.md).

## Customize a campaign

Use [guided setup](setup_wizard.md) to resolve model, dataset, and site
settings, and [configuration and overrides](configuration_overrides.md) for
persistent or temporary changes. Among the tracked routes, attention and GDN
reductions are enabled only by `vlm_campaign.yaml`; unsupported distributed and
packed modes fail closed. Inspect every customized plan with `--dry-run` before
launch.
