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
Treat SLURM walltime timeouts as expected resume events until `nel status`/`nel
info`, artifacts, and logs show a terminal failure or invalid run. Request or
task/trial timeouts require separate accounting below, even when the job succeeds.

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
6. If reasoning traces are present, confirm they are parsed/stripped/ignored before scoring consistently. Assess output-limit termination such as `finish_reason: length` using the accounting below; it is not by itself a parsing failure. Check for parser errors, unmatched reasoning delimiters, reasoning text leaked into answers, answers stripped with the reasoning, or reasoning disabled when the config intended it to be active.
7. Complete the **Timeout and Output-Limit Accounting** below for every task,
   including non-reasoning models and successful runs.

Report the run-validation summary before any score: log scan status, sample
accounting, reasoning/answer parsing status, and any errors or warnings found.
If any validation item fails, either rerun/fix it or label the result as
incomplete or invalid.

## Timeout and Output-Limit Accounting

Before reporting a score, inspect structured per-response and per-trial artifacts,
plus relevant logs and the resolved config. A successful job or MLflow export does
not establish that every sample finished without hitting a limit.

For each benchmark and each baseline/candidate run, report:

| Check | Required evidence |
|---|---|
| Coverage | Expected trials (including repeats), completed/scored trials, failed/skipped/missing trials, and the score denominator. State whether failures receive zero credit or are excluded. |
| Timeouts | Timed-out request attempts / all request attempts; unique trials affected / expected trials; and trials ending in timeout / expected trials. Separate recovered retries from terminal failures and identify the layer: client/proxy, agent/task, judge, or sandbox/verifier. |
| Output limits | Responses stopped by a generation limit / observed responses; unique affected trials / expected trials. Use explicit termination metadata such as `finish_reason: length`, and distinguish output-token caps from context exhaustion or agent step/total-token limits where evidence permits. |
| Effective limits | Request/proxy and task/agent/judge/verifier timeouts, timeout strategy, output-token and context limits, and any task overrides. Record concurrency, sharding, and serving setup alongside these limits. |

Show counts and percentages with explicit denominators and artifact paths. Count
unique sample/trial IDs including repeat IDs; deduplicate resumed artifacts and
keep request retries separate from trials. Categories can overlap, so do not sum
them as disjoint failures. Log keyword matches alone are not sample counts.
Token usage near a configured cap is a diagnostic clue, not proof of truncation;
inspect termination metadata and the effective per-request limit.

If artifacts omit termination reasons, cover only sampled responses, or exclude
failed requests, report that coverage and mark the full-run rate **unknown**.
Do not infer zero from missing telemetry or extrapolate a sampled rate to the
whole run. Identify the missing artifacts or instrumentation needed to resolve it.

Interpret the score under its actual protocol:

- Benchmark-defined time/token limits can legitimately produce failures; retain
  them in the official metric according to that protocol. Report the limit-hit
  rates rather than automatically declaring every nonzero rate invalid.
- A run may be **valid with warnings** when a small fraction of responses hit
  output limits or trials hit benchmark-defined timeouts, provided sample/repeat
  and scoring coverage are complete, limits match the intended protocol, and all
  other validation checks pass. Report the affected counts and any observed score
  impact; do not fail or automatically rerun solely because the rate is nonzero.
  Apply any explicit task/user tolerance. A low rate alone is not proof of
  negligible impact or a universal exemption for infrastructure failures, and
  run validity alone does not establish a quantization-feasibility verdict.
- Infrastructure failures, unexpected exclusions, or mismatched limits require
  investigation before a model-quality verdict. Unknown accounting makes this
  validation incomplete; a score may be shown as provisional, not validated.
- Matching wall-clock limits alone does not isolate model quality: serving speed,
  queueing, concurrency, and verbosity can change how much work fits in the limit.
  Compare limit-hit rates on both sides before attributing a delta to quantization;
  unresolved timeout effects make that attribution inconclusive.
- Do not silently drop timed-out trials or increase limits for only one model.
  If a controlled rerun is needed, use matched settings for both sides, preserve
  the original results, and label changes to the benchmark protocol explicitly.
  A longer-timeout diagnostic is not automatically a leaderboard-comparable score.

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
