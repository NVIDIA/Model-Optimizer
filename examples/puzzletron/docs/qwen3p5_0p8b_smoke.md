# Qwen 3.5 0.8B text pruning smoke

The checked-in `full_smoke` recipe runs a small end-to-end test of text-only
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

## Run and resume the text workflow

Use the checked-in `full_smoke.yaml` experiment and
`execution.full_smoke.yaml` execution config as the canonical inputs. The
checked-in `runner.slurm.yaml` is a portable template, not a runnable site
configuration. Copy it to a site-specific location once and replace its
`REPLACE_WITH_` values before launching. Dry-run accepts the portable template
for plan inspection, but the orchestrator rejects unresolved placeholders
before submitting work.

```bash
EXPERIMENT=examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/full_smoke.yaml
EXECUTION=examples/puzzletron/configs/orchestration/qwen3p5_0p8b/execution.full_smoke.yaml
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

The checked-in flow deliberately uses two candidate-evaluation samples, two
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

## Run the opt-in quality regression

`quality_regression.yaml` keeps the lifecycle smoke unchanged and adds a
non-default quality tier. It evaluates the complete IFEval and GSM8K task
datasets for both the final distilled student and the pinned teacher with the
same greedy-decoding contract. The result publishes candidate, reference, and
candidate-minus-reference metrics. Strict filters fail the DAG if either
accuracy regression exceeds its checked-in margin.

The fixed batch size of eight keeps the complete datasets inside the bounded
evaluation timeout. Treat batch size as part of the regression contract: even
with greedy decoding, changing it can alter some generated outputs.
Repeated runs with the same contract can still differ by a small number of
outputs because of backend numerical variation, so acceptance uses checked-in
metric margins rather than bit-for-bit equality.

Use
`execution.quality_regression.yaml` with a site-specific runner whose walltime
covers both serial evaluations. Set a distinct `PUZZLETRON_RUN_ROOT`; the full
task run is intentionally not part of default CI:

```bash
EXPERIMENT=examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/quality_regression.yaml
EXECUTION=examples/puzzletron/configs/orchestration/qwen3p5_0p8b/execution.quality_regression.yaml
RUNNER=/path/to/site-specific/runner.slurm.yaml
export PUZZLETRON_RUN_ROOT=/path/to/qwen3p5_0p8b_quality_regression

python examples/puzzletron/orchestrate.py \
  --experiment "$EXPERIMENT" \
  --runner "$RUNNER" \
  --execution "$EXECUTION" \
  --stage full --dry-run
```

After inspecting the plan, omit `--dry-run` to launch or resume it. Keep the
checkpoint revision, evaluator revision, task list, seed, and generation
settings fixed when comparing runs. The tiny model remains appropriate for
pipeline exception coverage, but its floor-level task scores are not a useful
quality-regression oracle.
