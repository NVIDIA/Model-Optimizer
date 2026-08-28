# Qwen 3.5 0.8B full-lifecycle smoke

The checked-in Qwen 3.5 0.8B `full_smoke` recipe pins the public checkpoint revision and
searches only the FFN intermediate sizes `[3072, 2048]`. The checked-in
`full_smoke.yaml` experiment extends the bounded MIP smoke through evaluation,
physical materialization, AIPerf, two global-distillation steps, final
evaluation, and final selection.

## Run and resume the full lifecycle

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

The checked-in flow deliberately uses two evaluation samples, four AIPerf
requests, and two distillation steps. These budgets validate lifecycle
correctness and resumability; they are not quality or throughput claims. Final
acceptance must reload the selected checkpoint, verify the cumulative report,
and confirm that the resume submits no work for completed stages.
