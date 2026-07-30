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

## External Baseline Sanity Check

For a baseline-vs-candidate comparison, perform this check after run validation
and before applying the candidate-delta gate or issuing a success verdict. This
is additional to, not a replacement for, the apples-to-apples and baseline
precision checks in `compare-results`.

For each baseline task:

1. Search for a published score for the exact model in its Hugging Face model
   card and on Artificial Analysis (<https://artificialanalysis.ai/>); use
   either credible source. A score for a sibling size, release, or precision is
   not an exact-model reference.
2. Match model variant, benchmark and version, metric, reasoning/thinking mode,
   prompt and chat template, sampling and token budget, sample count, and
   evaluation protocol as closely as possible. Record the external score,
   source URL, and every known protocol difference. Do not treat a mismatched
   result as directly comparable; find a closer source or mark the task
   externally unverified.
3. Put both scores on a 0-100 scale, then calculate, for higher-is-better
   metrics:

   ```text
   difference (pp) = abs(measured baseline - external)
   ```

   Treat a credible comparable result as verified only when the absolute
   difference is approximately 5 percentage points or less. A difference
   greater than approximately 5 points fails the check even if the candidate is
   within its normal delta gate (for example, `<1pp`). For an external score of
   60, the approximate range is 55-65; a baseline of 54 is 6 points away and
   fails.

Report each task as `verified`, `failed`, or `externally unverified`. A large
upward difference also does not establish a clean match; investigate protocol
differences before marking it verified. If no credible comparable score exists,
state `externally unverified` and do not invent a reference or claim the sanity
check passed. This status does not block comparison or publication: use the
validated measured baseline, apply the candidate-delta gate, and report that no
external corroboration was available. Only a `failed` external check blocks the
comparison.

If any task fails, do not report the quantized evaluation as successful and do
not apply the candidate-delta gate to that baseline. Investigate disabled
reasoning/thinking, reasoning parser or adapter handling, prompt/chat-template
differences, sampling or token-budget differences, benchmark version or metric,
incomplete samples, and serving failures. Rerun a corrected baseline, validate
it, and repeat this check before comparing the candidate.

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
