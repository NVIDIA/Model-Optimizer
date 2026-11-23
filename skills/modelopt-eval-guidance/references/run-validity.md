# Run Validity

Use this after a run finishes and before any score is compared, for every tool. Whether the run is healthy and
complete (logs, coverage, judge and sandbox faults, reasoning parsing, recovery) is decided by the evaluation
tool: take its verdict from the tool's reference. A tool with no verdict of its own lists the checks it needs in
its reference ("Run checks"). This file holds only what the quantized-versus-baseline comparison adds.

## Before any score is compared

- **The verdict decides eligibility.** A run whose verdict is not a pass, or whose coverage is incomplete or
  unverified, is labeled incomplete or invalid, its findings go to the parent or user, and no delta gate is applied
  to it. Do not resubmit a completed run on your own.
- **Keep score, validation and MLflow delivery separate**, and carry the counts and warnings into the handoff.

## Timeout and output-limit accounting

Take these counts from the tool's own run summary and report them for **both sides** before the score:
limit-stopped responses over observed responses, repetition stops ([doom loops](doom-loops.md)) counted
separately, timed-out trials over expected trials, and the coverage denominator. A rising rate on the quantized
side is a quality signal, so it belongs in the comparison even when the tool counts those responses as the
model's own result.

- **Unknown is not zero.** Missing termination metadata or an omitted failure list makes the rate unknown: say
  what evidence is missing and do not extrapolate.
- **Comparison:** matching wall-clock limits does not isolate model quality, since speed, queueing, concurrency
  and verbosity change the work done. Compare both sides' limit-hit rates; an unresolved timeout effect makes
  attribution to quantization inconclusive.
- **Reruns:** never silently drop timed-out trials or raise only one side's limits. Use matched settings, keep the
  original results, label the protocol change, and get the parent's or user's approval first.
- **Keep protocol limits fixed.** Report a prompt, history or output that does not fit the deployed context.

## What counts as an evaluation fault is the tool's rule

Apply the tool's own rule (its reference says where it is), and only to that tool's runs. Do not add a response-
level invalidation of your own: a wrong, truncated or empty answer, or a clean agent timeout, is the model's result
where the tool says so.

## Reading results from MLflow

Prefer a run id from the handoff; the tags and metric names each tool writes are in its reference.

- **Use a run only when its status is `FINISHED` and the metric exists.** One run is one benchmark of one
  submission: combine equal-repeat runs by averaging (two 16-repeat runs make a 32-repeat score), and take the task
  status from the tool's verdict, because a run can be `FINISHED` while its submission failed overall.
- Secondary metrics can aggregate differently from the headline (SciCode `subtask_accuracy`); compare only the
  headline.
