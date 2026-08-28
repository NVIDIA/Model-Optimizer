# Qwen 3.5 0.8B text pruning smoke

The checked-in `full_smoke` recipe runs a small end-to-end test of text-only
pruning for Qwen 3.5 0.8B. It searches the FFN intermediate sizes
`[3072, 2048]`, evaluates the candidates, saves the two strongest candidates as
physical checkpoints, and measures their serving performance with AIPerf. It
then distills the candidate with higher measured output-token throughput for
two steps, evaluates it again, and selects the final checkpoint. The recipe
pins the public checkpoint revision so repeated runs use the same starting
model.

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

The checked-in flow deliberately uses two evaluation samples per candidate,
four AIPerf requests per serving candidate, and two distillation steps. These
budgets validate lifecycle correctness, comparative serving selection, and
resumability; they are not quality or throughput claims. Final acceptance must
reload the selected checkpoint, verify the cumulative report, and confirm that
the resume submits no work for completed stages.
