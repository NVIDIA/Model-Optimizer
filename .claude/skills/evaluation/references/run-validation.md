# Run Validation

Use this reference when checking NEL progress after submission, resuming from
timeouts, validating completed runs, or handing completed baseline/candidate runs
to `compare-results`.

## NEL Timeout and Resume Behavior

NEL submissions commonly create a dependency chain of SLURM jobs. The first job
runs the evaluation and writes response/result caches. A dependent follow-on job
resumes from those caches if the first job times out, then queues another
follow-on job so long-running evals can continue across walltime windows.

Do not assume a timeout means the evaluation failed or produced invalid results.
Treat timeouts as expected resume events until `nel status`/`nel info`,
artifacts, and logs show a terminal failure or invalid run.

## Verify Completed Evaluation Run

Before pulling/reporting scores, validate the completed run itself. Do not
accept a run as complete just because `results.yml` or a summary file exists.

For each completed invocation/run directory, whether baseline, quantized, or a
single-model run:

1. Inspect client, server/deployment, SLURM, judge, and task-specific/code-execution logs as applicable. Search for `Traceback`, `Exception`, `ERROR`, `FAILED`, `OOM`, `Killed`, `timeout`, `rate limit`, `unauthorized`, `connection refused/reset`, `health check`, `sandbox`, `container`, `judge`, `parse`, `scoring`, and task-specific failure strings.
2. Confirm the inference server loaded the intended checkpoint/model and stayed healthy through the run: no startup failure, mid-run crash/restart, OOM, request validation failure, max-context truncation, quantization load error, or repeated 4xx/5xx responses.
3. For judge-backed tasks, confirm judge calls succeeded and were parsed/scored correctly: no auth/rate-limit failures, malformed judge responses, invalid JSON, missing scores, or fallback/default scores.
4. For code-execution tasks, inspect executor/sandbox/container logs for setup failures, package install failures, timeouts, thread/process exhaustion, permission errors, harness crashes, or skipped tests that would make scores non-comparable.
5. Confirm sample accounting: expected samples/repeats match completed, scored samples; no unexpected dropped/skipped/failed samples, `unknown_agent_error`, `failed_samples_policy` aborts, empty outputs, or partial result files.
6. If reasoning traces are present, confirm they are parsed/stripped/ignored before scoring consistently. Check for parser errors, unmatched reasoning delimiters, `finish_reason: length`, reasoning text leaked into answers, answers stripped with the reasoning, or reasoning disabled when the config intended it to be active.

Report the run-validation summary before any score: log scan status, sample
accounting, reasoning/answer parsing status, and any errors or warnings found.
If any validation item fails, either rerun/fix it or label the result as
incomplete or invalid.

For score harvesting, use the `Score Extraction` section from the matching task
reference in `recipes/tasks/<task>.md`. Do not rely on ad hoc `results.yml`
greps when a task reference defines the canonical score and stderr fields.

For baseline-vs-candidate deltas, use the `compare-results` skill after each run
passes validation.

## NEL Diagnostics

```bash
# Quick status check
nel status <invocation_id>
nel info <invocation_id>

# Get log paths
nel info <invocation_id> --logs

# Inspect logs via SSH
ssh <user>@<host> "tail -100 <log_path>/server-<slurm_job_id>-*.log"   # deployment errors
ssh <user>@<host> "tail -100 <log_path>/client-<slurm_job_id>.log"     # evaluation errors
ssh <user>@<host> "tail -100 <log_path>/slurm-<slurm_job_id>.log"      # scheduling/walltime
ssh <user>@<host> "grep -i 'traceback\|exception\|error\|failed\|oom\|killed\|timeout\|unauthorized\|rate limit\|sandbox\|container\|judge\|parse\|scoring' <log_path>/*.log"  # search all logs
```
