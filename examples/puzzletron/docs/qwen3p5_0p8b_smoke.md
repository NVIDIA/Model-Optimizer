# Qwen 3.5 0.8B text pruning smoke

The `full_smoke` recipe runs a small end-to-end test of text-only
pruning for Qwen 3.5 0.8B. It searches the FFN intermediate sizes
`[3072, 2048]`, evaluates the candidates, saves the two strongest candidates as
physical checkpoints, and reloads each saved directory through vLLM for two
IFEval samples. It then measures both checkpoints with AIPerf, distills the
candidate with higher measured output-token throughput for two steps, evaluates
the resulting checkpoint with another two IFEval samples and the internal
two-sample LM-loss check, and selects the final checkpoint. The recipe pins the
public checkpoint revision so repeated runs use the same starting model. See
[evaluate saved checkpoints](post_mip_pipeline.md#evaluate-saved-checkpoints)
for how both Hugging Face directories are loaded without an AnyModel-to-AutoModel
conversion.

These small budgets check that the complete workflow runs and resumes
correctly. They do not establish model quality or production throughput.

## Generate a complete bundle with the setup wizard

For a new run, start with the [setup wizard](setup_wizard.md) and select Qwen
3.5 0.8B. Its model-specific defaults generate both smoke and production
bundles covering conversion, pruning, search, MIP selection, materialization,
serving measurement, short distillation, final selection, and a pinned
student-versus-teacher quality comparison. Inspect the generated
`dry-run-plan.txt` and materialize the site-specific runner settings before
launching. The comparison is measurement-only and does not accept the current
scores as a quality baseline.

The tracked recipes below remain useful as reviewable reference configurations
and for reproducing the bounded GPU comparison.

## Run and resume the text workflow

Use the `full_smoke.yaml` experiment with the shared `execution.single_gpu.yaml`
profile. The `runner.slurm.yaml` file is a portable template, not a
runnable site configuration. Copy it to a site-specific location and replace its
`REPLACE_WITH_` values before launching. Dry-run accepts the portable template
for plan inspection, but the orchestrator rejects unresolved placeholders
before submitting work.

```bash
EXPERIMENT=examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/full_smoke.yaml
EXECUTION=examples/puzzletron/configs/orchestration/execution.single_gpu.yaml
RUNNER=/path/to/site-specific/runner.slurm.yaml
export PUZZLETRON_RUN_ROOT=/path/to/qwen3p5_0p8b_full_smoke
```

Inspect the complete one-GPU plan without submitting work:

```bash
python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full --dry-run
```

After reviewing the dry-run, launch with the same three inputs and omit only
`--dry-run`:

```bash
python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full
```

Resume by rerunning that exact launch command with the same experiment,
materialized runner, and execution config:

```bash
python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full
```

The flow deliberately uses two candidate-evaluation samples, two
IFEval samples, four AIPerf requests per serving candidate, and two
distillation steps. These budgets validate workflow correctness, comparative
serving selection, and resumability; they are not quality or throughput claims.
The worker environment must provide the [pinned evaluator
installation](checkpoint_evaluation.md#quick-start). IFEval task data must be
fetchable from each worker or already present in its Hugging Face cache.

After completion, inspect the `checkpoint_eval` and `post_kd_checkpoint_eval`
nodes under `artifacts/post_mip/nodes`. Their summaries must name the corresponding
pre-KD and post-KD checkpoints, report two effective IFEval samples, and contain
finite metrics. Also verify the cumulative report and confirm that resuming
submits no work for completed stages.

## Run the opt-in end-to-end quality comparison

`e2e_quality_comparison.yaml` inherits the full smoke workflow and adds a
bounded quality comparison. It evaluates the same pinned 100-example IFEval
and GSM8K subsets for both the final distilled student and the pinned teacher
with the same greedy-decoding settings. The result publishes student, teacher,
and student-minus-teacher metrics without applying a pass/fail threshold. The
same evaluation is also used by the
[Qwen 3.5 0.8B campaign](qwen3p5_0p8b_campaign.md), so the affordable comparison
exercises its final downstream evaluation without repeating the larger search.

Two repeated GPU runs produced identical measurements: IFEval was 0.23 for the
student and 0.55 for the teacher, while GSM8K was 0.01 for the student and 0.45
for the teacher. The evaluated `params-90` student retained about 89.96% of the
checkpoint parameters, or about 10.04% parameter pruning. Its realized
architecture reduced every one of the 24 FFN intermediate widths from 3584 to
2048, a 42.86% reduction within those FFN dimensions. These results establish
the comparison mechanism, not an acceptable quality baseline. The observation
combines this FFN reduction with only two distillation optimizer steps, so it
does not isolate the effect of pruning and does not test whether a longer,
tuned distillation phase can recover quality. Establish and validate a stronger
pruning and distillation recipe before adding regression thresholds.

These historical measurements are stored only in the opt-in comparison recipe.
They are not setup defaults and are not emitted by the wizard or inherited by
the larger campaign.

The evaluator commit, dataset revisions, first 100 rows, seed, generation
settings, and batch size are fixed. Keep batch size 8 when comparing runs.
Backend numerical variation can still change a small number of outputs, so
compare the reported metrics and logged samples rather than expecting
bit-for-bit equality. Each fresh comparison also publishes
`observation_delta.*` metrics and records `difference_from_recorded` in
`comparison.json` for the four headline scores above. These differences are
diagnostic values only; no tolerance or pass/fail rule is applied.

Use the shared `execution.single_gpu.yaml` profile with a site-specific runner
whose walltime covers both serial evaluations. Set a distinct
`PUZZLETRON_RUN_ROOT`; the comparison is intentionally not part of default CI:

```bash
EXPERIMENT=examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/e2e_quality_comparison.yaml
EXECUTION=examples/puzzletron/configs/orchestration/execution.single_gpu.yaml
RUNNER=/path/to/site-specific/runner.slurm.yaml
export PUZZLETRON_RUN_ROOT=/path/to/qwen3p5_0p8b_e2e_quality_comparison

python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full --dry-run
```

After inspecting the plan, omit `--dry-run` to launch or resume it. Keep the
checkpoint revision, evaluator revision, task list, seed, and generation
settings fixed when comparing runs. The tiny model is useful for exercising
the pipeline and checking student-to-teacher score changes, but its absolute
task scores should not be treated as a quality target.
