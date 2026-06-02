---
name: eagle3-review-logs
description: >
  Review EAGLE3 pipeline experiment logs from the launcher's experiments/ directory.
  Summarizes pass/fail status for all 4 tasks, diagnoses failures with root causes
  and fixes, and flags warnings. Use when the user asks to review job logs,
  check experiment results, or diagnose why a specific task failed.
user_invocable: true
---

# Review EAGLE3 Experiment Logs

Analyze output logs from an EAGLE3 pipeline run launched via `launch.py` or `slurm.py`.

## Step 0 — Find experiment logs

Locate the experiment directory. The default is `experiments/` relative to the launcher root,
or wherever `--job-dir` was pointed.

```bash
ls -td experiments/cicd/cicd_* | head -10
```

Each experiment has one subdirectory per task (0–3). Logs are `sbatch_*.out` files inside:

```bash
find experiments/<exp_id>/ -name "sbatch_*.out" | sort
```

Do this in a single Bash call. If no experiments exist, ask the user for the directory.

## Step 1 — Read all task logs

Read the last 200 lines of each log in parallel. Errors appear at the end:

```bash
for f in $(find experiments/<exp_id>/ -name "sbatch_*.out" | sort); do
  echo "=== $f ==="; tail -200 "$f"; echo
done
```

## Step 2 — Analyze

For each task log, check:

- **Exit / cancellation**: `DUE TO TIME LIMIT`, `FAILED`, signal (e.g., `signal 15`)
- **Python exceptions / tracebacks**: last exception is usually the root cause
- **CUDA errors**: OOM, NCCL timeout
- **Slurm state**: COMPLETED, FAILED, TIMEOUT, OUT_OF_MEMORY
- **Success indicators**: "Saved N samples", "Successfully processed N conversations", training loss line, AR output

## Step 3 — Produce report

Output a structured markdown report:

### Summary

- Overall status: PASSED / FAILED / MIXED / PARTIAL
- Task breakdown: e.g., task_0 TIMEOUT, task_1 FAIL, task_2 skipped, task_3 skipped

### Task Results

For each task (0–3):

**Task N — \<name\>: PASS / FAIL / TIMEOUT**
- Key output: (e.g., "3277/3295 samples generated" or "Script not found")
- Error (if failed): quoted error message, max 10 lines
- Root cause: one-line diagnosis
- Suggested fix: actionable step

### Warnings

Non-fatal issues worth noting (near-OOM, tokenizer warnings, slow throughput).

## Step 4 — Suggest next steps

Based on results:

- If a task failed due to a known issue, suggest the fix and how to re-run from that task:

  ```bash
  uv run launch.py --yaml examples/<Org>/<Model>/hf_offline_eagle3.yaml \
      pipeline.task_0.skip=true \
      --yes
  ```

- If the failure pattern is new (not in `tools/launcher/examples/EAGLE3_TRIAGE.md`),
  suggest adding it to the triage chart using `/eagle3-triage` guidance.

- If all tasks passed, suggest running `/eagle3-validate` to confirm AR meets threshold.

## Known benign patterns (do NOT mark as failures)

| Pattern | Explanation |
|---|---|
| vLLM server exit code 143 | SIGTERM — server was killed after queries completed. Expected. |
| `CANCELLED AT ... DUE TO TASK FAILURE` after `exit code: 0` | Slurm cleanup of worker nodes after main task succeeded. |
| `destroy_process_group() was not called` | Benign PyTorch shutdown warning. |
| `tokenizer class ... not equal to the registered tokenizer class` | Harmless tokenizer mismatch warning. |
