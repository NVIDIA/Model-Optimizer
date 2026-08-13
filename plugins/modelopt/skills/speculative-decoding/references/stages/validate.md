# Stage 4 — Validate a completed run

Verify that a pipeline run completed successfully end-to-end and meets its quality
gate.

## Step 0 — Identify the experiment

Find the most recent experiment directory (or ask the user for the path):

```bash
ls -td experiments/cicd/cicd_* | head -5
```

Each experiment directory has one subdirectory per task, each containing a log file
whose name varies by launch mode (Slurm: `sbatch_*.out`, local Docker: `*.log`).

## Step 1 — Check task outcomes

Match the log files generally and read the tail of each:

```bash
find experiments/<exp_id>/ -type f \( -name '*.out' -o -name '*.log' \) | sort | while read -r f; do
  echo "=== $f ==="; tail -50 "$f"; echo
done
```

Every task must complete without error. Look for:

- `exit code: 0` or no error — success
- `DUE TO TIME LIMIT` — timeout
- `FAILED` / `signal` / exception traceback — failure

If any task failed, go to `triage.md` instead.

## Step 2 — Verify artifacts exist

Check each task produced its expected output. Artifacts live on the cluster under
`/scratchspace/`, so confirm via log messages. The per-task log evidence and artifact
paths are in *Success markers* in `../algorithms/<algorithm>.md`.

## Step 3 — Check the quality gate

*Quality gate* in `../algorithms/<algorithm>.md` gives the metric, the log line it
appears on, and the pass threshold. Extract the value from the benchmark task's log
and compare.

If the log already reports the metric below its lower bound, the run tripped the
threshold check itself and exited non-zero.

## Step 4 — Check training quality

In the training task's log look for:

- **Final training loss** — should be decreasing, not NaN
- **Metric validation during training** — if the recipe enabled periodic validation
- **Number of training steps** — confirms full training duration

## Step 5 — Produce validation report

```markdown
## Speculative Decoding Pipeline Validation Report

**Experiment:** <exp_dir>
**Model:** <model_name>
**Algorithm:** <algorithm>
**Date:** <date>
**Pipeline config:** <yaml_path>

### Task Status
| Task | Name | Status | Notes |
|------|------|--------|-------|
| 0 | Data synthesis | PASS/FAIL/TIMEOUT | N samples generated |
| 1 | Hidden state dump | PASS/FAIL | N .pt files |
| 2 | Training + export | PASS/FAIL | Final loss: X.XX |
| 3 | Benchmark | PASS/FAIL | AR: X.XX |

### Quality Gate
- <metric>: X.XX (threshold: <threshold>) — PASS/FAIL

### Training Summary
- Final loss: X.XX
- Training steps: N
- Metric during training: X.XX (if validated)

### Overall: PASS / FAIL
<one-line summary>
```

Adjust the task rows to the tasks this config actually ran — task count varies by
algorithm and variant.

## Step 6 — Suggest next steps

**If PASS:**

- Record the verified result (and checkpoint path) in the team's internal triage
  tracker
- This model is now a candidate to add as a launcher example in a dedicated PR

**If FAIL:**

- Identify which task or metric failed
- Go to `triage.md` for diagnosis
- For a low acceptance rate, diagnose the specific cause from the run (training loss
  curve, data volume/quality, draft capacity, hyperparameters) and suggest fixes
  targeted to that scenario — a low rate can have many causes, so avoid a generic
  checklist.
