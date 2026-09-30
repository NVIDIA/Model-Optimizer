# Check progress of a running evaluation

Follow the three phases and track your progress in the output.

1. **INPUT** -> EXPLORE -> ACT
2. ~~INPUT~~ -> **EXPLORE** -> ACT
3. ~~INPUT~~ -> ~~EXPLORE~~ -> **ACT**

## 1. INPUT

- **Invocation ID**: The evaluation to monitor. After a `nel run` submission, this is printed in the output as `Invocation ID: <id>`.

## 2. EXPLORE

1. **Get status & task name**: `uv run nemo-evaluator-launcher status <invocation_id> --json`
2. **Check for benchmark-specific docs**: Read files in `references/benchmarks/` matching the task name (e.g., `terminal-bench-general-info.md` for `terminal-bench-*` tasks). These contain monitoring commands and benchmark-specific context.
3. **Get output paths from config**: `uv run nemo-evaluator-launcher info <invocation_id>` → find `output_dir` and cluster hostname.

## 3. ACT

1. Report status, slurm job ID, task name from step 2.1
2. **If RUNNING**: SSH to cluster and check the live progress in the `client-*.log` file. Use the monitoring command from benchmark docs if exists.
3. **If SUCCESS**: Pivot to analyzing results. See `references/analyze-results.md`.
4. **If FAILED**: Pivot to debugging failed runs. See `references/debug-failed-runs.md`.
