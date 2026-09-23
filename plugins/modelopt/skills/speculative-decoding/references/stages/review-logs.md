# Stage 2 — Review experiment logs

Analyze output logs from a pipeline run launched via `launch.py` or `slurm.py`, and
produce a pass/fail summary across all tasks. For a deep dive into one failing task,
go to `triage.md` instead.

## Step 0 — Find the experiment

The default job directory is `experiments/` relative to the launcher root, or
wherever `--job-dir` was pointed.

```bash
ls -td experiments/cicd/cicd_* | head -10
```

If no experiments exist, ask the user for the directory.

## Step 1 — Read all task logs

Each experiment has one subdirectory per task. Log filenames vary by launch mode
(Slurm writes `sbatch_*.out`, local Docker writes `*.log`), so match log files
generally and read the tail of each in a single Bash call — errors surface at the end:

```bash
find experiments/<exp_id>/ -type f \( -name '*.out' -o -name '*.log' \) | sort | while read -r f; do
  echo "=== $f ==="; tail -200 "$f"; echo
done
```

## Step 2 — Analyze

For each task log, check:

- **Exit / cancellation**: `DUE TO TIME LIMIT`, `FAILED`, signal (e.g. `signal 15`)
- **Python exceptions / tracebacks**: the last exception is usually the root cause
- **CUDA errors**: OOM, NCCL timeout
- **Slurm state**: COMPLETED, FAILED, TIMEOUT, OUT_OF_MEMORY
- **Success indicators**: see *Success markers* in `../algorithms/<algorithm>.md` —
  each task has a specific log line that proves it worked

## Step 3 — Produce report

Output a structured markdown report:

### Summary

- Overall status: PASSED / FAILED / MIXED / PARTIAL
- Task breakdown: e.g. task_0 TIMEOUT, task_1 FAIL, task_2 skipped, task_3 skipped

### Task Results

For each task:

**Task N — \<name\>: PASS / FAIL / TIMEOUT**

- Key output: (e.g. "3277/3295 samples generated" or "Script not found")
- Error (if failed): quoted error message, max 10 lines
- Root cause: one-line diagnosis
- Suggested fix: actionable step

### Warnings

Non-fatal issues worth noting (near-OOM, tokenizer warnings, slow throughput).

## Step 4 — Suggest next steps

- If a task failed due to a known issue, suggest the fix and how to re-run from that
  task:

  ```bash
  uv run launch.py --yaml examples/<Org>/<Model>/<config>.yaml \
      pipeline.task_0.skip=true \
      --yes
  ```

- If the failure pattern looks new, suggest capturing it in the team's internal
  triage tracker, and use `triage.md` for a deeper diagnosis.
- If all tasks passed, move to `validate.md` to confirm the quality gate.

## Known benign patterns (do NOT mark as failures)

| Pattern | Explanation |
| --- | --- |
| vLLM server exit code 143 | SIGTERM — server was killed after queries completed. Expected. |
| `CANCELLED AT ... DUE TO TASK FAILURE` after `exit code: 0` | Slurm cleanup of worker nodes after the main task succeeded. |
| `destroy_process_group() was not called` | Benign PyTorch shutdown warning. |
| `tokenizer class ... not equal to the registered tokenizer class` | Harmless tokenizer mismatch warning. |
