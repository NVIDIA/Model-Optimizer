---
name: eagle3-validate
description: >
  Validate that an EAGLE3 pipeline run completed successfully end-to-end.
  Checks all 4 steps produced expected artifacts, verifies acceptance rate
  meets threshold (>= 2.1), and produces a summary report.
  Use when user wants to verify a pipeline run or check benchmark results.
user_invocable: true
---

# EAGLE3 Pipeline Validation

Verify that an EAGLE3 pipeline run completed successfully and meets quality criteria.

## Step 0 — Identify the experiment

Find the most recent experiment directory (or ask the user for the path):

```bash
ls -td experiments/cicd/cicd_* | head -5
```

Each experiment directory has one subdirectory per task (numbered 0–3), each containing
a `sbatch_*.out` log file.

## Step 1 — Check task outcomes

Read the last 50 lines of each task's log file:

```bash
find experiments/<exp_id>/ -name "sbatch_*.out" | sort | while read f; do
  echo "=== $f ==="; tail -50 "$f"; echo
done
```

All 4 tasks must complete without error. Look for:
- `exit code: 0` or no error — success
- `DUE TO TIME LIMIT` — timeout
- `FAILED` / `signal` / exception traceback — failure

If any task failed, suggest running `/eagle3-triage` instead.

## Step 2 — Verify artifacts exist

Check each step produced the expected output (artifacts live on the cluster at `/scratchspace/`).
Confirm via log messages:

| Step | Expected log evidence | Artifact |
|------|-----------------------|----------|
| task_0 | "Saved N samples" or progress bar completing | `/scratchspace/data/*.jsonl` |
| task_1 | "Successfully processed N conversations" | `/scratchspace/offline_hidden_states/*.pt` |
| task_2 | Training loss decreasing, "export complete" | `/scratchspace/eagle3/model.safetensors`, `/scratchspace/export/` |
| task_3 | `Average Acceptance Length ... ratio: X.XX` | JSON result files |

## Step 3 — Check acceptance rate

In the task_3 log, find:

```
Average Acceptance Length {'accept': X, 'count': Y, 'ratio': Z.ZZ}
```

The `ratio` field is the acceptance rate (AR).

| Criterion | Threshold | Status |
|-----------|-----------|--------|
| AR (MT-Bench) | >= 2.1 | PASS / FAIL |

If the log shows `AR ... < lower bound`, the run already triggered a threshold failure (exit code 1).

## Step 4 — Check training quality

In the task_2 log look for:
- **Final training loss** — should be decreasing, not NaN
- **AR validation during training** (if `training.ar_validate_steps` was set)
- **Number of training steps** — confirms full training duration

## Step 5 — Produce validation report

```markdown
## EAGLE3 Pipeline Validation Report

**Experiment:** <exp_dir>
**Model:** <model_name>
**Date:** <date>
**Pipeline config:** <yaml_path>

### Step Status
| Step | Task | Status | Notes |
|------|------|--------|-------|
| 0 | Data synthesis | PASS/FAIL/TIMEOUT | N samples generated |
| 1 | Hidden state dump | PASS/FAIL | N .pt files |
| 2 | Training + export | PASS/FAIL | Final loss: X.XX |
| 3 | Benchmark | PASS/FAIL | AR: X.XX |

### Acceptance Rate
- MT-Bench AR: X.XX (threshold: >= 2.1) — PASS/FAIL

### Training Summary
- Final loss: X.XX
- Training steps: N
- AR during training: X.XX (if validated)

### Overall: PASS / FAIL
<one-line summary>
```

## Step 6 — Suggest next steps

**If PASS:**
- The model's row in `tools/launcher/examples/EAGLE3_TRIAGE.md` can be updated to ✅
- Note the checkpoint path for downstream use

**If FAIL:**
- Identify which step or metric failed
- Suggest running `/eagle3-triage` for diagnosis
- If AR is close but below threshold, suggest:
  - More training epochs (`training.num_epochs` override)
  - More training data (re-run task_0 with larger dataset)
  - Larger draft head (`num_hidden_layers` in `eagle_config.json`)
  - Hyperparameter tuning (`training.lr`, `training.train_bs`)
