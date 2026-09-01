# Qwen 3.5 0.8B text campaign

Use this example to run a Qwen 3.5 0.8B text pruning campaign. It searches FFN
intermediate sizes 3328 and 3072 while retaining teacher values for the other
architecture dimensions. The campaign scores replacements, selects
parameter-constrained MIP solutions, runs 128 KD optimizer steps for each
candidate, and ranks the candidates using LM and downstream metrics. It then
runs 256 KD optimizer steps for the selected candidate from its materialized
pre-KD checkpoint and evaluates that student against the teacher on the full
downstream task splits.

The campaign uses larger scoring, validation, serving, and distillation budgets
than the smoke recipes. The resulting student depends on the pruning target and
KD configuration.

## Inspect the complete plan

Use the campaign experiment and execution files with a site-specific runner:

```bash
EXPERIMENT=examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/campaign.yaml
EXECUTION=examples/puzzletron/configs/orchestration/qwen3p5_0p8b/execution.campaign.yaml
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

The default campaign changes only `ffn.intermediate_size`. The opt-in
`campaign_extended.yaml` variant additionally enables hidden width,
grouped-attention KV and query heads, GDN key groups, GDN value-head dimension,
embedding width, and depth. Each enabled axis requires compatible measurement
and checkpoint-slicing support.

Run the extended variant with the same two-GPU KD execution profile:

```bash
EXPERIMENT=examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/campaign_extended.yaml
EXECUTION=examples/puzzletron/configs/orchestration/qwen3p5_0p8b/execution.campaign.yaml
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
shows the configuration shape for the additional Qwen axes. Include only axes
whose measurement and slicing implementations support the selected checkpoint.
Compile the resulting campaign with `--dry-run`, then run its smoke recipe
before using the full campaign budgets.

## Interpret the final quality comparison

The screening quality step evaluates both distilled candidates on fixed
256-example prefixes of pinned IFEval, GSM8K, MMLU-Pro computer science, and
MMLU-Pro history revisions. The final quality step evaluates the selected
student and teacher on the complete task splits using the same evaluator
version, generation settings, seed, batch size, and task definitions. Final
results contain student, teacher, and student-minus-teacher metrics plus
generated samples for qualitative inspection.
The [opt-in quality comparison](qwen3p5_0p8b_smoke.md#run-the-opt-in-end-to-end-quality-comparison)
uses smaller search and distillation budgets with the same downstream
evaluation settings. Neither route defines quality acceptance thresholds.
