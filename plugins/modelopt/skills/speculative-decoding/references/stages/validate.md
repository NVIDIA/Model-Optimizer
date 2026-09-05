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

Check each task produced its expected output. The per-task log evidence and artifact
paths are in *Success markers* in `../algorithms/<algorithm>.md`.

A success line in a log is not proof the artifact survived — the next task reads it
from a shared `/scratchspace`, where it may be missing, empty, or unreadable. **When
you can reach the cluster, check the filesystem directly** and treat a missing or
zero-byte artifact as a validation failure:

```bash
test -s <artifact_path> && echo "ok: $(du -sh <artifact_path>)" || echo "MISSING/EMPTY"
ls -la <artifact_dir>/ | head
```

Fall back to log evidence only when the cluster isn't reachable, and say so in the
report rather than implying the artifacts were verified.

## Step 3 — Check the quality gate

Read *Quality gate* in `../algorithms/<algorithm>.md` first — **not every algorithm
produces an in-pipeline metric**, so what you check depends on the sheet:

- **Sheet defines a benchmark metric** (e.g. EAGLE3's MT-Bench AR, DFlash's
  `Average_AL`) — extract it from the benchmark task's log and compare against the
  threshold. If the log already reports the metric below its lower bound, the run
  tripped the threshold check itself and exited non-zero.
- **Sheet defines no inference metric** (currently Domino and DSpark — their eval path
  runs the DFlash backbone with the new head bypassed, and Domino ships no benchmark
  task at all) — do **not** go looking for a benchmark log. Report the training
  regression gate instead, and state plainly that acceptance quality requires a
  separate evaluation of the exported checkpoint. Never report a backbone-only
  acceptance rate as the model's result.

Where the gate is the training regression check (`check_regression.py` against
`trainer_state.json`), confirm the `=== Regression Check ===` block is actually
present in the log — it is invoked with `|| true` and only warns when no
`trainer_state.json` exists, so a green exit does not prove it ran.

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
