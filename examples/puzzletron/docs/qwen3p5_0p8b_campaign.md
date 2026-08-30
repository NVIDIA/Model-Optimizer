# Qwen 3.5 0.8B text campaign

Use this example when you want to run a substantial Qwen 3.5 0.8B text pruning
campaign rather than a small workflow test. It searches FFN width, hidden
width, grouped attention, GDN, embedding, and depth choices; learns bypass
candidates; selects a serving-aware MIP solution; distills the selected
student; and finishes with a student-versus-teacher downstream evaluation.

The campaign uses larger scoring, validation, serving, and distillation budgets
than the smoke recipes. These choices demonstrate a broad workflow, but they
are starting points rather than prescribed settings. Adapt the search space,
constraints, workloads, data, and training budget to the deployment target.
Complete a smoke run before allocating resources to the larger campaign.

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

## Interpret the final quality comparison

The final quality step evaluates the distilled student and pinned teacher on
the first 100 examples of pinned IFEval and GSM8K revisions. Both models use the
same evaluator version, greedy generation settings, seed, batch size, and task
definitions. The result contains student, teacher, and student-minus-teacher
metrics plus generated samples for qualitative inspection.

The [opt-in quality regression](qwen3p5_0p8b_smoke.md#run-the-opt-in-end-to-end-quality-regression)
uses smaller search and distillation budgets, reuses the same downstream
evaluation settings, and adds calibrated acceptance gates. It exercises the
campaign's final evaluation without repeating the larger search.
