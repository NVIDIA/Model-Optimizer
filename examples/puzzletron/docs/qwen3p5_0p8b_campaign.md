# Qwen 3.5 0.8B text campaign

Use this example to run a Qwen 3.5 0.8B text pruning campaign. It searches the
conservative FFN intermediate sizes 3328 and 3072. Other architecture
dimensions remain at their teacher values. The campaign scores replacements,
selects parameter-constrained MIP solutions, gives both candidates a 128-step
KD screen, ranks their LM and downstream metrics, and gives the winner a clean
256-step KD run from its materialized pre-KD checkpoint. It finishes with a
full student-versus-teacher downstream evaluation.

The campaign uses larger scoring, validation, serving, and distillation budgets
than the smoke recipes. Complete a smoke run before allocating resources to
this campaign. The checked-in search and distillation parameters are starting
values for exercising the route, not settings optimized for downstream quality.
Tune the pruning amount and distillation duration together before using the
resulting scores as expected model quality.

## Inspect the complete plan

Use the campaign experiment and execution files with a site-specific runner:

```bash
EXPERIMENT=examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/campaign.yaml
EXECUTION=examples/puzzletron/configs/orchestration/execution.two_gpu_kd.yaml
RUNNER=/path/to/site-specific/runner.yaml
export PUZZLETRON_RUN_ROOT=/path/to/qwen3p5_0p8b_campaign
export PUZZLETRON_DATASET_PATH=/path/to/puzzle_kd_dataset

python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full --dry-run
```

Inspect every requested stage, node count, walltime, and worker path before
removing `--dry-run`. Relaunch the same command to resume compatible work;
completed stages are not submitted again.

## Change the search dimensions

The default Qwen 3.5 0.8B campaign and its real-checkpoint validation cover
only `ffn.intermediate_size`. The opt-in `campaign_extended.yaml` variant
also exposes hidden width, grouped-attention KV and query heads, GDN key groups,
GDN value-head dimension, embedding width, and depth. These dimensions are
available for additional searches, but have less real-checkpoint coverage and
may need model-specific fixes.

Run the extended variant with the same two-GPU KD execution profile:

```bash
EXPERIMENT=examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/campaign_extended.yaml
EXECUTION=examples/puzzletron/configs/orchestration/execution.two_gpu_kd.yaml
```

To create another variant, copy a run config and make each new dimension
explicit in three places:

1. Declare its legal candidate values in the model config.
2. Enable its measurement, sanity, and replacement-scoring stages in the run
   config.
3. Add the measured axis to each applicable `mip.runs.<name>.search_space`.

Keep `axes_default: teacher` so omitted dimensions retain the teacher value.
The [configuration guide](configuration_overrides.md) explains where model and
run settings live, and [MIP profiles](mip_profiles.md#search-space) documents
the search-space syntax. The
[advanced Qwen overlay](../configs/families/qwen3_5/qwen3p5_0p8b/advanced.yaml)
shows the configuration shape for additional Qwen axes; use those axes only
after validating their measurements and slicing behavior for the selected
checkpoint. Run the resulting campaign with `--dry-run`, then exercise it as a
smoke before increasing its budgets.

## Interpret the final quality comparison

The screening quality step evaluates both distilled candidates and the pinned
teacher on fixed 256-example prefixes of pinned IFEval, GSM8K, MMLU-Pro computer
science, and MMLU-Pro history revisions. The final quality step evaluates the
winning student and teacher on the complete task splits. All comparisons use
the same evaluator version, greedy generation settings, seed, batch size, and
task definitions. Results contain student, teacher, and student-minus-teacher
metrics plus generated samples for qualitative inspection.
The [opt-in quality comparison](qwen3p5_0p8b_smoke.md#run-the-opt-in-end-to-end-quality-comparison)
uses smaller search and distillation budgets and reuses the same downstream
evaluation settings. Its checkpoint-specific historical observation is not
inherited by this campaign, and neither route defines acceptance thresholds.
