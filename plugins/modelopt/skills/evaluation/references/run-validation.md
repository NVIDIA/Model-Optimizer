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
2. Verify the intended checkpoint loaded; inspect startup failures, crashes/restarts, OOMs, request errors, input/context clipping, and quantization load errors. Classify attributable per-trial failures under the policy below; systemic serving failures remain blockers.
3. For judge-backed tasks, inspect authentication, rate limits, malformed responses, parsing, and scoring. Distinguish protocol-valid failed scores from missing scores or undocumented fallback/default scores.
4. For code-execution tasks, inspect executor/sandbox/container setup, installation, timeout, resource, permission, harness, and skipped-test diagnostics. Verify each failed trial's protocol-defined outcome rather than treating every error as invalid.
5. Confirm coverage at every available level: expected, selected, accounted-for, and scored samples/repeats/trajectories must match. Explicit protocol-valid failed trials count as accounted-for; missing/unscored trials, dropped submissions, `failed_samples_policy` aborts leaving gaps, and partial results do not.
6. If reasoning traces are present, confirm they are parsed/stripped/ignored before scoring consistently. Assess unusable model outputs under the policy below. Check for parser or fallback errors, unmatched reasoning delimiters, reasoning text leaked into answers, answers stripped with the reasoning, or reasoning disabled when the config intended it to be active.
7. Complete the **Timeout and Output-Limit Accounting** below for every task,
   including non-reasoning models and successful runs.

Report the run-validation summary before any score: log scan status, coverage,
reasoning/answer parsing status, and any errors or warnings found. If an
independent validation item fails, label the result incomplete or invalid and
return findings and a recommendation to the parent (or user); do not
automatically resubmit a completed run.

### Bounded Evaluation-Failure Policy (Parent and Evaluator)

For each benchmark/run, report category counts, their deduplicated union, the
verified denominators, and `rate = 100 × union / denominator` for each applicable
gate below. Calculate with unrounded counts; round only the displayed percentage.

**Response gate (model-output faults).** Preserve unique-response accounting:

The denominator is the deduplicated set of unique, successful raw evaluated-model
responses selected for evaluation. Count a cached response reused by multiple
trajectories once; exclude duplicate log/cache records, failed request attempts,
judge calls, and unrelated runs. Verify the denominator from response identity
and provenance rather than dataset size, especially for repeated or multi-turn
tasks. An unknown or zero denominator leaves the rate unverified, not 0%.

The numerator is the set union of denominator responses unusable because of
model output behavior:

- length or token-budget truncation, including reasoning that consumes the budget;
- an empty final answer, including a nonempty reasoning trace with no final answer;
- a malformed or otherwise unusable final output.

Count a response in every applicable category for reporting, but once in the
union. To qualify, the successful raw response must exist, remain preserved in
the artifacts, and be deterministically retained and scored incorrect. A parser
exception, parser fallback/default score, or missing parsed record is not
automatically a model-output fault. A malformed answer qualifies only if the
preserved raw response is explicitly retained and scored incorrect under the
benchmark protocol. Runtime scorer/harness failures require trial-gate assessment
and protocol-valid scoring instead.

**Trial gate (runtime failures).** Use the expected benchmark trial/repeat count,
not successful trials, HTTP attempts, or the number of records found. Its numerator
is the deduplicated union of attributable terminal request/transport/server,
judge, executor/sandbox/verifier, terminal action, solver, and harness failures,
including timeouts, that the benchmark explicitly records and scores incorrect
or zero under its existing protocol. A failed request without a raw response may
qualify here only through such a scored trial; it cannot enter the response gate.
Ordinary wrong answers are not runtime failures.

Preserve diagnostics and original raw evidence privately, including failed
submissions. Verify failure type and scoring from structured artifacts and logs;
an exception string or a summary zero alone is insufficient. Deduplicate overlaps
and resumes by invocation, benchmark, task, trial, and repeat identity. Count
overlapping categories once within each gate; never add response and trial rates
or pool benchmarks. Both applicable gates must pass. A shared fault may affect
both gates, but is counted once within each. Mark an inapplicable gate explicitly;
an unknown applicable denominator is not an exemption.

Continue other trials after a harness crash only when the harness can record the
affected trial as a protocol-valid scored failure. If it cannot, report a blocker.
Do not fabricate zeros, modify scoring semantics, treat missing/unscored trials
as completed, select passing repetitions, or silently regrade old runs. Keep the
original trial and score denominators; omitted zero-scored failures leave coverage
incomplete. Never average incomplete data.

- **rate = 0%:** no failure warning for that gate; independent checks still apply.
- **0 < rate ≤ 2.0% (inclusive, before rounding):** valid with a visible warning
  only when coverage is complete and all independent gates pass. Retain failures
  and their protocol-defined scores; do not invalidate, abort the invocation, or
  retry solely for bounded failures. A low rate does **not** imply negligible
  score impact or leaderboard comparability.
- **rate > 2.0%:** return category counts, union, rate, findings, and a
  recommendation to the parent (or user); do not automatically retry or declare
  success.
- The tolerance never waives secret leaks (scanning remains fail-closed), wrong
  model/task/version/configuration, falsified or provenance-less output,
  cross-invocation contamination, input/context clipping, systemic serving or
  broken scoring, incomplete coverage, or benchmark-specific validity rules.
  These are independent integrity failures, not ordinary per-trial failures.
  Undocumented parser/judge fallback scores remain invalid. A score above a
  reference does not establish validity.
- Check configured and effective output limits against the reference evaluation
  protocol and deployed context capacity, including prompt/history plus output
  space. Report mismatches or unavailable reference settings. Do not remove or
  tune token limits merely to pass; propose protocol-justified changes for
  parent/user approval instead.

Keep score, run validation, and MLflow delivery separate: a valid score does not
prove export succeeded, and export cannot validate scoring. In a day0 run summary,
keep policy-validated failure diagnostics in `warnings`; `errors` contains
unresolved blockers and still fails `gate_run.py`. That summary gate does not
compute these rates or verify raw evidence; perform this policy check first.
The parent must retain both gates' counts, rates, and warnings in handoffs.

## Timeout and Output-Limit Accounting

For every benchmark/run, including successful and non-reasoning runs, inspect
structured response/trial artifacts, logs, and resolved config. Report counts,
percentages, explicit denominators, and artifact paths before the score:

| Check | Required evidence |
|---|---|
| Coverage | Expected trials including repeats; completed/scored, failed/skipped/missing; score denominator and whether failures receive zero or are excluded. |
| Timeouts | Timed-out / all request attempts; affected unique trials / expected trials; terminal-timeout trials / expected trials. Separate recovered retries and terminal failures by layer: client/proxy, agent/task, judge, sandbox/verifier. |
| Output limits | Limit-stopped / observed responses; affected unique trials / expected trials. Use termination metadata such as `finish_reason: length`; distinguish output caps, context exhaustion, and agent step/total-token limits where possible. |
| Limits | Effective request/proxy, agent/task, judge/verifier, output-token and context limits, timeout strategy, overrides, concurrency, sharding, and serving setup. |

Deduplicate resumed records by sample/trial/repeat ID; keep request attempts
separate from trials. Categories overlap, so do not sum them as disjoint failures.
Log matches and token counts near a cap are clues, not proof of affected samples.
Missing termination metadata, sampled-only artifacts, or omitted failures make
full-run rates **unknown**, not zero. Report coverage and what evidence is missing;
do not extrapolate sampled rates.

- **Terminal failures:** apply the bounded policy above to output faults and
  protocol-valid scored runtime/trial failures, including harness timeouts.
  Report timeout type, effective limits, concurrency/queueing, and protocol
  differences; tolerance alone does not establish comparability.
- **Recovered retries:** report them separately. They do not enter the fault
  numerator or denominator as extra responses, and they do not excuse a terminal
  unscored failure or missing coverage.
- **Provisional/inconclusive:** unknown accounting leaves validation incomplete.
  Investigate infrastructure failures, unexpected exclusions, or mismatched
  limits before a model-quality verdict. A valid run alone does not establish
  quantization feasibility.
- **Comparison:** matching wall-clock limits does not isolate model quality;
  speed, queueing, concurrency, and verbosity affect work completed. Compare
  both sides' limit-hit rates; unresolved timeout effects make attribution to
  quantization inconclusive.
- **Reruns:** never silently drop timed-out trials or increase only one model's
  limits. Use matched settings, preserve original results, and label protocol
  changes. Longer-timeout diagnostics are not automatically leaderboard-comparable.

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
