# Run Validity

Use this after a run finishes and before any score is reported, for every tool. The tool's own health
verdict is the primary signal; these are the checks and the reporting that do not depend on the tool.

How much applies depends on the tool. A tool with no run verdict of its own needs every check below. A tool that
already decides state, health, coverage and recovery (its reference says whether it does) gives a verdict to take
for "Verify a completed run" instead of repeating the log scan; keep the accounting, the comparison rules and the
reporting, which belong to the comparison.

## Verify a completed run

Do not accept a run as complete because a result file or summary exists. For each completed baseline,
quantized or single-model run:

1. **Logs.** Inspect client, server or deployment, scheduler, judge and task or code-execution logs as they
   apply. Search for `Traceback`, `Exception`, `ERROR`, `FAILED`, `OOM`, `Killed`, `timeout`, `rate limit`,
   `unauthorized`, `connection refused/reset`, `health check`, `sandbox`, `container`, `judge`, `parse` and
   `scoring`, plus task-specific failure strings.
2. **The intended checkpoint loaded.** Inspect startup failures, crashes and restarts, OOMs, request errors,
   input or context clipping, and quantization load errors. Systemic serving failures are blockers.
3. **Judge-backed tasks.** Inspect authentication, rate limits, malformed responses, parsing and scoring.
   Distinguish a protocol-valid failed score from a missing score or an undocumented fallback score.
4. **Code-execution tasks.** Inspect executor, sandbox and container setup, installation, timeout, resource,
   permission and harness diagnostics. Verify each failed trial's protocol-defined outcome; do not treat every
   error as invalid.
5. **Coverage at every level.** Expected, selected, accounted-for and scored samples, repeats or trajectories
   must match. A protocol-valid failed trial counts as accounted for; a missing or unscored trial, a dropped
   submission, an abort that leaves gaps, or a partial result does not.
6. **Reasoning parsing.** If reasoning traces are present, confirm they are parsed, stripped or ignored before
   scoring, consistently. Look for parser or fallback errors, unmatched reasoning delimiters, reasoning text
   leaked into answers, answers stripped with the reasoning, or reasoning disabled when the config meant it on.
7. **Accounting.** Complete the section below for every task, including successful and non-reasoning runs.

Report a validation summary before any score: log scan status, coverage, reasoning and answer parsing status,
and any errors or warnings. If a check fails, label the result incomplete or invalid and return the findings
and a recommendation to the parent or user; do not resubmit a completed run on your own. Keep score,
validation and MLflow delivery separate, and carry counts, rates and warnings into the handoff.

## Timeout and output-limit accounting

For every benchmark and run, inspect structured response or trial artifacts, logs and the resolved config.
Report counts, percentages, explicit denominators and artifact paths before the score:

| Check | Required evidence |
|---|---|
| Coverage | Expected trials including repeats; completed and scored, failed, skipped, missing; the score denominator and whether failures get zero or are excluded. |
| Timeouts | Timed-out over all request attempts; affected unique trials over expected trials; terminal-timeout trials over expected trials. Separate recovered retries from terminal failures by layer: client or proxy, agent or task, judge, sandbox or verifier. |
| Output limits | Limit-stopped over observed responses; affected unique trials over expected trials. Use termination metadata such as `finish_reason: length` (and [doom-loop](doom-loops.md) repetition stops, counted separately); separate output caps, context exhaustion, and agent step or total-token limits where possible. |
| Limits | Effective request, agent and judge limits, output-token and context limits, timeout strategy, overrides, concurrency, sharding and serving setup. |

Deduplicate resumed records by sample, trial and repeat id, and keep request attempts separate from trials.
The categories overlap, so do not sum them as disjoint failures. Log matches and token counts near a cap are
clues, not proof of affected samples. Missing termination metadata, sampled-only artifacts or omitted failures
make full-run rates **unknown**, not zero: say what evidence is missing and do not extrapolate.

- **Recovered retries** are reported separately; they do not excuse a terminal unscored failure or missing
  coverage.
- **Provisional or inconclusive:** unknown accounting leaves validation incomplete. Investigate infrastructure
  failures, unexpected exclusions and mismatched limits before a model-quality verdict. A valid run alone does
  not establish quantization feasibility.
- **Comparison:** matching wall-clock limits does not isolate model quality, since speed, queueing, concurrency
  and verbosity change the work completed. Compare both sides' limit-hit rates; unresolved timeout effects make
  attribution to quantization inconclusive.
- **Reruns:** never silently drop timed-out trials or raise only one model's limits. Use matched settings,
  preserve the original results, and label protocol changes.
- **Keep protocol limits fixed.** Verify that prompt, history and output fit the deployed context, and report
  mismatches or unknown reference settings; proposed changes need the parent's or user's approval.

## What counts as an evaluation fault is the tool's rule

Apply the tool's own rule (its reference says where it is and which health verdict to take), and only to that
tool's runs. Do not add a response-level invalidation of your own: a wrong, truncated or empty answer, or a clean
agent timeout, is the model's result where the tool says so. Still report the per-side truncation and output-limit
counts above, because a rising rate on the quantized side is a quality signal for the comparison.

## Reading results from MLflow

Every supported tool exports to the same MLflow experiment, so prefer a run id from the handoff and read the run
there. The tags and metric names each tool writes are in its reference (checked on finished runs, read-only).

- **Use a run only when its status is `FINISHED` and the metric exists.** Failed and killed runs are exported
  too, with few metrics. One run is one benchmark of one submission: combine equal-repeat runs by averaging
  (two 16-repeat runs make a 32-repeat score), and take the task status from the tool's own verdict, because a
  benchmark's run can be `FINISHED` while its submission failed overall.
- Secondary metrics can aggregate differently from the headline (SciCode `subtask_accuracy`); compare only the
  headline.
